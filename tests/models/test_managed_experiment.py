from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dryml.artifacts import CachedDataset, PARQUET_KIND, PARQUET_REPRESENTATION
from dryml.artifacts.representations.parquet import write_parquet_sequence
from dryml.core2 import Repo
from dryml.core2.object import Pickleable
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedCapabilityError,
    ManagedOutput,
    MissingManagedOutputError,
    OperationPreflight,
    StaleManagedResultError,
    current_operation_context,
    managed,
    transfer_realizations,
)
from dryml.managed.dispatch import ManagedDispatchRequest
from dryml.records import ExecutionRecord, StoredStateRecord
from dryml.models import Experiment, Model, TrainFunction
from dryml.models.utils import advance_train_state


class ScalarModel(Model, Pickleable):
    def __init__(self, initial=0):
        self.value = initial


class SumTraining(TrainFunction):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, exp):
        rows = [int(np.asarray(row)[0]) for row in exp.train_data]
        exp.model.value += self.scale * sum(rows)
        advance_train_state(exp, epochs=1, steps=len(rows))
        return exp.model.value


class CountingArrayDataset(ArrayDataset):
    builds = 0

    def __init__(self, arrays, *, spec=None, batched=True, validate_lengths=True):
        type(self).builds += 1
        super().__init__(
            arrays,
            spec=spec,
            batched=batched,
            validate_lengths=validate_lengths,
        )


class ParquetCachedDataset(CachedDataset):
    def __dryml_managed_preflight__(self, method, args, kwargs):
        if method != "compute" or args or kwargs:
            raise TypeError("invalid ParquetCachedDataset compute request")
        return OperationPreflight(False, None, False)

    def __dryml_managed_validate_invocation__(self, *args, **kwargs):
        pass

    @managed(
        outputs=(
            ManagedOutput(
                "data",
                primary=True,
                kind="data",
                representations=(PARQUET_KIND,),
            ),
        ),
        resumable=False,
        early_completion=False,
    )
    def compute(self):
        context = current_operation_context()
        source = Repo(context.store).load_or_build(self.src)
        with tempfile.TemporaryDirectory(prefix="dryml-test-parquet-cache-") as temp:
            root = Path(temp)
            write_parquet_sequence(source, root, partition_rows=2)
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                context.write_output(
                    "data",
                    path.relative_to(root).as_posix(),
                    (path.read_bytes(),),
                    representation=PARQUET_REPRESENTATION,
                    subject_cdef_id=format_cdef_id(self.definition.stable_hash()),
                )


def _cache(store, values):
    cached = CachedDataset(ArrayDataset(np.asarray(values, dtype=np.int64)[:, None]))
    cached.compute(store=store, representation="numpy-sequence", shard_rows=2)
    return cached


def _execution(store, result):
    realization = store.records.read_record(result.realization_record_id)
    execution_id = realization["payload"]["execution_record_id"]
    return ExecutionRecord.from_envelope(store.records.read_record(execution_id))


def _direct_consumed_ids(execution):
    return tuple(
        link.record_id
        for link in execution.consumed_records
        if link.producer_cdef_id is None
    )


def _control_snapshot(store):
    root = Path(store.managed_control_root())
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _mismatched_fine_tune(store):
    producer = Experiment(
        ScalarModel(1),
        SumTraining(),
        train_data=_cache(store, [2]),
    )
    producer.train(store=store)
    consumer = Experiment(
        ScalarModel(10),
        SumTraining(),
        train_data=_cache(store, [3]),
        model_state=producer.train.result,
    )
    return consumer


def test_experiment_identity_includes_model_train_function_and_all_data():
    model = ScalarModel()
    trainer = SumTraining()
    train = CachedDataset(ArrayDataset(np.array([[1]], dtype=np.int64)))
    val = CachedDataset(ArrayDataset(np.array([[2]], dtype=np.int64)))
    test = CachedDataset(ArrayDataset(np.array([[3]], dtype=np.int64)))

    baseline = Experiment(model, trainer, train_data=train, val_data=val, test_data=test)

    variants = (
        Experiment(ScalarModel(1), trainer, train_data=train, val_data=val, test_data=test),
        Experiment(model, SumTraining(2), train_data=train, val_data=val, test_data=test),
        Experiment(model, trainer, train_data=CachedDataset(ArrayDataset(np.array([[4]], dtype=np.int64))), val_data=val, test_data=test),
        Experiment(model, trainer, train_data=train, val_data=CachedDataset(ArrayDataset(np.array([[5]], dtype=np.int64))), test_data=test),
        Experiment(model, trainer, train_data=train, val_data=val, test_data=CachedDataset(ArrayDataset(np.array([[6]], dtype=np.int64)))),
    )

    assert all(item.definition != baseline.definition for item in variants)


def test_train_outputs_are_delegated_without_materializing_train_function():
    exp = Experiment(
        ScalarModel(),
        SumTraining(),
        train_data=CachedDataset(ArrayDataset(np.array([[1]], dtype=np.int64))),
    )

    outputs = type(exp).__dict__["train"].output_declarations(exp.definition)

    assert outputs.slots == ("model",)
    assert outputs.primary.kind == "stored_state"
    assert outputs.primary.subject_path == ("model",)
    assert exp.train.result.slot == "model"


def test_training_rejects_missing_completed_cache_without_computing_source(tmp_path):
    store = DirStore(tmp_path / "store")
    source = CountingArrayDataset(np.array([[1], [2]], dtype=np.int64))
    cached = CachedDataset(source)
    exp = Experiment(ScalarModel(), SumTraining(), train_data=cached)
    CountingArrayDataset.builds = 0

    with pytest.raises(MissingManagedOutputError, match="active"):
        exp.train(store=store)

    assert CountingArrayDataset.builds == 0
    assert exp.train.status(store=store).status == "not_started"


def test_managed_training_snapshots_initial_state_and_hydrates_fresh_model(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    model = ScalarModel()
    repo.save_object(model, record_policy="none")
    cached = _cache(store, [1, 2, 3])
    exp = Experiment(model, SumTraining(), train_data=cached)

    result = exp.train(store=store)
    execution = _execution(store, result)
    direct_ids = _direct_consumed_ids(execution)
    trained_record = StoredStateRecord.from_envelope(
        store.records.read_record(result.outputs["model"].record_id)
    )

    assert len(result.consumed_records) == 1
    assert result.consumed_records[0].record_id == cached.compute.results(store=store)["data"].record_id
    assert len(direct_ids) == 1
    assert StoredStateRecord.from_envelope(store.records.read_record(direct_ids[0])).state_role == "initial-model-state"
    assert trained_record.subject_cdef_id == format_cdef_id(model.definition.stable_hash())

    first = exp.trained_model(store=store)
    second = exp.trained_model(store=store)
    assert first is not second
    assert first is not model
    assert first.value == second.value == 6
    assert model.value == 0


def test_definition_only_model_uses_fresh_initial_state(tmp_path):
    store = DirStore(tmp_path / "store")
    cached = _cache(store, [1, 2])
    exp = Experiment(ScalarModel(5), SumTraining(), train_data=cached)
    Repo(store).save_definition(exp.definition)

    exp.train(store=store)

    assert exp.trained_model(store=store).value == 8


def test_mutated_ordinary_model_state_changes_consumed_snapshot(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    model = ScalarModel()
    repo.save_object(model, record_policy="none")
    exp = Experiment(model, SumTraining(), train_data=_cache(store, [1]))

    first = exp.train(store=store)
    first_snapshot = _direct_consumed_ids(_execution(store, first))[0]
    model.value = 10
    repo.save_object(model, record_policy="none")

    with pytest.raises(StaleManagedResultError):
        exp.train(store=store)

    second = exp.train.rerun(store=store)
    second_snapshot = _direct_consumed_ids(_execution(store, second))[0]
    assert second_snapshot != first_snapshot
    assert exp.trained_model(store=store).value == 11


def test_shared_model_definition_has_experiment_scoped_state_and_fine_tune_lineage(tmp_path):
    store = DirStore(tmp_path / "store")
    model = ScalarModel()
    Repo(store).save_object(model, record_policy="none")
    first = Experiment(model, SumTraining(), train_data=_cache(store, [1, 2]))
    second = Experiment(model, SumTraining(), train_data=_cache(store, [10]))

    first_result = first.train(store=store)
    second_result = second.train(store=store)

    assert first_result.outputs["model"].record_id != second_result.outputs["model"].record_id
    assert first.trained_model(store=store).value == 3
    assert second.trained_model(store=store).value == 10

    fine_data = _cache(store, [4])
    fine_tune = Experiment(
        model,
        SumTraining(),
        train_data=fine_data,
        model_state=first.train.result,
    )
    fine_result = fine_tune.train(store=store)

    assert fine_tune.trained_model(store=store).value == 7
    assert tuple(item.record_id for item in fine_result.consumed_records) == (
        fine_data.compute.results(store=store)["data"].record_id,
        first_result.outputs["model"].record_id,
    )


def test_fine_tune_subject_mismatch_fails_before_local_control_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    consumer = _mismatched_fine_tune(store)
    before = _control_snapshot(store)

    with pytest.raises(ManagedCapabilityError, match="model_state.*subject"):
        consumer.train(store=store)

    assert _control_snapshot(store) == before
    assert consumer.train.status(store=store).status == "not_started"
    assert consumer.train.history(store=store) == ()


def test_fine_tune_subject_mismatch_fails_before_dispatch_control_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    consumer = _mismatched_fine_tune(store)
    before = _control_snapshot(store)
    request = ManagedDispatchRequest(consumer.train, (), {})
    session = None

    try:
        with pytest.raises(ManagedCapabilityError, match="model_state.*subject"):
            session = request._prepare(SimpleNamespace(store=store))
    finally:
        if session is not None:
            session.lease.release()

    assert _control_snapshot(store) == before
    assert consumer.train.status(store=store).status == "not_started"
    assert consumer.train.history(store=store) == ()


def test_managed_training_consumes_completed_parquet_cache_without_rebuild(tmp_path):
    pytest.importorskip("pyarrow")
    store = DirStore(tmp_path / "store")
    source = CountingArrayDataset(np.array([[2], [3]], dtype=np.int64))
    cached = ParquetCachedDataset(source)
    cached_result = cached.compute(store=store)
    CountingArrayDataset.builds = 0
    exp = Experiment(ScalarModel(), SumTraining(), train_data=cached)

    result = exp.train(store=store)

    assert result.consumed_records[0].record_id == cached_result.outputs["data"].record_id
    assert CountingArrayDataset.builds == 0
    assert exp.trained_model(store=store).value == 5


def test_trained_experiment_transfer_includes_direct_initial_state(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    model = ScalarModel(5)
    Repo(source).save_object(model, record_policy="none")
    exp = Experiment(model, SumTraining(), train_data=_cache(source, [1, 2]))
    result = exp.train(store=source)
    initial_state_id = _direct_consumed_ids(_execution(source, result))[0]

    report = transfer_realizations(source, destination, exp.train.result)
    reopened = Repo(destination).load(exp.definition, restore_state=False)

    assert initial_state_id in report.records
    assert destination.records.read_record(initial_state_id)["kind"] == "stored_state"
    assert reopened.trained_model(store=destination).value == 8


def test_completed_cache_is_pinned_without_source_rebuild_during_training(tmp_path):
    store = DirStore(tmp_path / "store")
    source = CountingArrayDataset(np.array([[2], [3]], dtype=np.int64))
    cached = CachedDataset(source)
    cached.compute(store=store, representation="numpy-sequence")
    CountingArrayDataset.builds = 0

    exp = Experiment(ScalarModel(), SumTraining(), train_data=cached)
    exp.train(store=store)

    assert CountingArrayDataset.builds == 0
    assert exp.trained_model(store=store).value == 5


def test_dispatched_training_uses_existing_managed_method_path(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    exp = Experiment(ScalarModel(), SumTraining(), train_data=_cache(store, [2, 4]))
    environment = PythonExecutableSpec(
        sys.executable,
        pythonpath_policy="explicit",
        extra_pythonpath=(str(Path(__file__).parent),),
    ).to_data()

    result = Dispatcher(store=store).run(
        exp.train,
        environment=environment,
        timeout=20,
    )

    assert result.status == "ok", result.error
    assert result.managed_result["action"] == "start"
    assert len(result.managed_result["consumed_records"]) == 1
    assert exp.trained_model(store=store).value == 6
