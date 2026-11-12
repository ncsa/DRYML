"""Checkpoint history discovery remains coherent while other workers publish."""

import multiprocessing
import traceback

from dryml.core import Definition, Par, Repo
from dryml.core.query.source import _NodePage
from dryml.core.store.dir import DirStore
from dryml.models import Experiment, ExperimentData
from tests.models.test_experiment_artifacts import CounterModel, SavedModelValue, TerminalOnly


class CountedTraining(TerminalOnly):
    """One terminal update with a process-local count of authored executions."""

    calls = 0

    def __call__(self, exp, *, callbacks=()):
        type(self).calls += 1
        return super().__call__(exp, callbacks=callbacks)


def test_history_discovery_recaptures_before_creating_or_adopting(tmp_path, monkeypatch, fixed_snapshot_environment):
    repo = Repo(DirStore(tmp_path / "store"))
    peer = Repo(DirStore.open_existing(repo.default_store.base_dir))
    try:
        subject = repo.save(CounterModel(1, repo=repo)).object_view()
        orphan = ExperimentData(subject, repo=peer)
        original = _NodePage.expand
        published = []

        def publish_during_lookup(page, *args, **kwargs):
            result = original(page, *args, **kwargs)
            if not published:
                published.append(True)
                peer.save(orphan)
            return result

        monkeypatch.setattr(_NodePage, "expand", publish_during_lookup)
        history = ExperimentData.get_or_create(subject, repo=repo)
        assert history.object_id == orphan.object_id and not history._rows
        current = ExperimentData.find(subject, repo=repo)
        assert current.object_id == orphan.object_id
        assert repo.query().saved().sel(ExperimentData._definition(subject)).count() == 1
    finally:
        peer.close(flush=False)
        repo.close(flush=False)


def test_current_history_name_retries_before_restoring_payload(tmp_path, monkeypatch, fixed_snapshot_environment):
    repo = Repo(DirStore(tmp_path / "store"))
    peer = Repo(DirStore.open_existing(repo.default_store.base_dir))
    try:
        subject = repo.save(CounterModel(1, repo=repo)).object_view()
        expected = ExperimentData.get_or_create(subject, repo=repo)
        name = repo.default_store.read_node_name(ExperimentData._current_name(subject))
        read = repo.default_store.read_node
        load = ExperimentData.restore_state_from_dir_imp
        moved, restored = [], []

        def read_and_move(digest, **kwargs):
            node = read(digest, **kwargs)
            if digest == name.root_digest and not moved:
                moved.append(True)
                peer.save(CounterModel(2, repo=peer))
            return node

        def counted_restore(obj, *args, **kwargs):
            restored.append(True)
            return load(obj, *args, **kwargs)

        monkeypatch.setattr(repo.default_store, "read_node", read_and_move)
        monkeypatch.setattr(ExperimentData, "restore_state_from_dir_imp", counted_restore)
        actual = ExperimentData.find(subject, repo=repo)
        assert actual.object_id == expected.object_id
        assert moved == restored == [True]
    finally:
        peer.close(flush=False)
        repo.close(flush=False)


def test_artifact_completion_buffers_candidates_before_control_lookup(
        tmp_path, monkeypatch, fixed_managed_snapshot_environment):
    from contextlib import closing
    from dryml.core.query.query import IdentityQuery
    from dryml.managed import ManagedConfig
    from tests.models.test_experiment_artifacts import ConstantValue

    repo = Repo(DirStore(tmp_path / "store"))
    peer = Repo(DirStore.open_existing(repo.default_store.base_dir))
    try:
        value = ConstantValue(repo=repo)
        repo.save(value)
        ConstantValue.calls = 0
        result = value.compute(managed=ManagedConfig(state_repo=repo))
        value._install_value_payload({"format": "dryml.artifacts.value", "version": 1, "present": True, "result": 9})
        repo.save(value)
        original = IdentityQuery._iter_terminal_members
        moved = []

        def change_after_first(query, *args, **kwargs):
            with closing(original(query, *args, **kwargs)) as items:
                for item in items:
                    yield item
                    if not moved:
                        moved.append(True)
                        peer.save(CounterModel(3, repo=peer))

        monkeypatch.setattr(IdentityQuery, "_iter_terminal_members", change_after_first)
        completed = ConstantValue.find_completed_state_ref(value, repo=repo)
        assert completed.graph_equal(result.final_state_ref)
        assert ConstantValue.calls == 1 and moved == [True]
    finally:
        peer.close(flush=False)
        repo.close(flush=False)


def test_history_name_readback_retries_without_republishing(tmp_path, monkeypatch, fixed_snapshot_environment):
    repo = Repo(DirStore(tmp_path / "store"))
    peer = Repo(DirStore.open_existing(repo.default_store.base_dir))
    try:
        subject = repo.save(CounterModel(1, repo=repo)).object_view()
        store = repo.default_store
        save, cas, read = repo.save_object, store.compare_and_set_node_name, store.read_node
        saves, committed, moved = [], [], []

        def counted_save(*args, **kwargs):
            saves.append(True)
            return save(*args, **kwargs)

        def counted_cas(*args, **kwargs):
            result = cas(*args, **kwargs)
            committed.append(result)
            return result

        def read_and_move(*args, **kwargs):
            node = read(*args, **kwargs)
            if committed and not moved:
                moved.append(True)
                peer.save(CounterModel(2, repo=peer))
            return node

        monkeypatch.setattr(repo, "save_object", counted_save)
        monkeypatch.setattr(store, "compare_and_set_node_name", counted_cas)
        monkeypatch.setattr(store, "read_node", read_and_move)
        history = ExperimentData.get_or_create(subject, repo=repo)
        assert len(committed) == 1 and saves == moved == [True]
        assert ExperimentData.find(subject, repo=repo).object_id == history.object_id
    finally:
        peer.close(flush=False)
        repo.close(flush=False)


def _train_with_history(root, value, barrier, result):
    from dryml.environments import introspection
    from dryml.managed import ManagedConfig
    from tests.conftest import _synthetic_environment_record

    introspection.inspect_current = _synthetic_environment_record
    repo = Repo(DirStore.open_existing(root))
    try:
        exp = Experiment(CounterModel(value), CountedTraining(),
            artifacts={"score": Definition(SavedModelValue, Par("this.model"))}, repo=repo)
        barrier.wait(timeout=30)
        invocation = exp.train(managed=ManagedConfig(state_repo=repo))
        result.send((invocation.execution.operation_id, CountedTraining.calls, len(SavedModelValue.calls)))
    except BaseException:
        result.send(traceback.format_exc())
    finally:
        repo.close(flush=False)
        result.close()


def test_concurrent_experiment_checkpoints_publish_complete_history_and_artifacts(tmp_path):
    from dryml.managed.control import ManagedControlStore

    repo = Repo(DirStore(tmp_path / "store"))
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    pipes = [ctx.Pipe(duplex=False) for _ in range(2)]
    processes = [ctx.Process(target=_train_with_history,
        args=(repo.default_store.base_dir, value, barrier, pipes[index][1]))
        for index, value in enumerate((32, 64))]
    try:
        for process in processes:
            process.start()
        outcomes = []
        for reader, writer in pipes:
            writer.close()
            assert reader.poll(120), "Experiment worker did not finish"
            outcomes.append(reader.recv())
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
        control = ManagedControlStore(repo.default_store, repo)
        for initial, outcome in zip((32, 64), outcomes):
            assert isinstance(outcome, tuple), outcome
            operation, trains, evaluations = outcome
            assert (trains, evaluations) == (1, 1)
            current = control.inspect(operation)
            assert current.state == "completed"
            restored = repo.load_state_ref(current.final, reuse_live="never")
            assert restored.model.value == initial + 1
            history = ExperimentData.find(current.final.object_view(), repo=repo)
            assert len(history._rows) == 1
            row = next(iter(history._rows.values()))
            assert row["evaluation_status"] == "completed"
            assert row["facts"]["state_ref"].graph_equal(current.final)
            assert dict(row["eval_artifacts"])["score"] is not None
        assert not repo.default_store.authority_cut().pending
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(10)
        for reader, writer in pipes:
            reader.close()
            writer.close()
        repo.close(flush=False)
