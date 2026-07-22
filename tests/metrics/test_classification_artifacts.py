from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core2 import Repo, TensorSpec
from dryml.core2.object import Pickleable
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.managed import MissingManagedOutputError, StaleManagedResultError
from dryml.metrics import CategoricalAccuracy, ConfusionMatrix
from dryml.metrics.classification import _normalize_observed_labels
from dryml.models import Experiment, Model, TrainFunction
from dryml.models.utils import advance_train_state
from dryml.records import DataRecord, RepresentationSpec


class CountingPairs(ArrayDataset):
    builds = 0

    def __init__(self, arrays, *, spec=None, batched=True, validate_lengths=True):
        type(self).builds += 1
        super().__init__(
            arrays,
            spec=spec,
            batched=batched,
            validate_lengths=validate_lengths,
        )


class TableClassifier(Model, Pickleable):
    builds = 0

    def __init__(self, scores):
        type(self).builds += 1
        self.scores = np.asarray(scores, dtype=np.float64)

    def __call__(self, x):
        return self.scores[np.asarray(x, dtype=np.int64)]


class SparseTableClassifier(Model, Pickleable):
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def __call__(self, x):
        return self.labels[np.asarray(x, dtype=np.int64)]


class NoOpTraining(TrainFunction):
    runs = 0

    def __call__(self, exp):
        type(self).runs += 1
        advance_train_state(exp, epochs=1, steps=sum(1 for _ in exp.train_data))


def _cache(store, x, y, *, source_type=CountingPairs):
    source = source_type((np.asarray(x), np.asarray(y)))
    cached = CachedDataset(source)
    cached.compute(store=store, representation="numpy-sequence", shard_rows=2)
    return cached


def _workflow(store, *, scores=None, y=None, model_type=TableClassifier):
    if scores is None:
        scores = (
            (0.9, 0.05, 0.05),
            (0.1, 0.2, 0.7),
            (0.1, 0.1, 0.8),
            (0.1, 0.8, 0.1),
        )
    if y is None:
        y = (0, 1, 2, 1)
    train = _cache(store, (0,), (0,))
    test = _cache(store, range(len(y)), y)
    model = model_type(scores)
    Repo(store).save_object(model, record_policy="none")
    experiment = Experiment(model, NoOpTraining(), train_data=train)
    trained = experiment.train(store=store)
    return model, experiment, test, trained


def _data_record(store, invocation):
    return DataRecord.from_envelope(
        store.records.read_record(invocation.outputs["metric"].record_id)
    )


def test_accuracy_and_confusion_publish_exact_record_backed_results(tmp_path):
    store = DirStore(tmp_path / "store")
    _model, experiment, test, trained = _workflow(store)
    accuracy = CategoricalAccuracy(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )
    confusion = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )

    accuracy_result = accuracy.compute(store=store)
    confusion_result = confusion.compute(store=store)

    assert accuracy.value(store=store) == pytest.approx(0.75)
    np.testing.assert_array_equal(
        confusion.matrix(store=store),
        np.array(((1, 0, 0), (0, 1, 1), (0, 0, 1)), dtype=np.int64),
    )
    assert accuracy_result.consumed_records[0].record_id == trained.outputs["model"].record_id
    assert accuracy_result.consumed_records[1].record_id == test.compute.results(store=store)["data"].record_id
    assert confusion_result.consumed_records == accuracy_result.consumed_records

    record = _data_record(store, confusion_result)
    representation = RepresentationSpec(
        store.records.read_spec(record.representation_id, family="representation")
    )
    assert record.realization_id == confusion_result.realization_id
    assert record.output_slot == "metric"
    assert representation.parameters == {
        "columns": "predicted",
        "labels": (0, 1, 2),
        "rows": "true",
    }


@pytest.mark.parametrize("one_hot_true", [False, True])
def test_sparse_and_one_hot_labels_and_predictions_normalize_when_unambiguous(
    tmp_path, one_hot_true
):
    store = DirStore(tmp_path / str(one_hot_true))
    labels = np.array((0, 1, 2, 1), dtype=np.int64)
    true_values = np.eye(3, dtype=np.int64)[labels] if one_hot_true else labels
    _model, experiment, test, _trained = _workflow(store, y=true_values)

    accuracy = CategoricalAccuracy(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
        batch_size=2,
    )
    confusion = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
        batch_size=2,
    )

    accuracy.compute(store=store)
    confusion.compute(store=store)

    assert accuracy.value(store=store) == pytest.approx(0.75)
    np.testing.assert_array_equal(
        confusion.matrix(store=store),
        ((1, 0, 0), (0, 1, 1), (0, 0, 1)),
    )


@pytest.mark.parametrize("batch_size", [None, 2])
def test_singleton_sparse_true_and_predicted_labels_are_unambiguous(
    tmp_path, batch_size
):
    store = DirStore(tmp_path / str(batch_size))
    predicted = np.array(((0,), (2,), (2,), (1,)), dtype=np.int64)
    true_values = np.array(((0,), (1,), (2,), (1,)), dtype=np.int64)
    _model, experiment, test, _trained = _workflow(
        store,
        scores=predicted,
        y=true_values,
        model_type=SparseTableClassifier,
    )
    accuracy = CategoricalAccuracy(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
        batch_size=batch_size,
    )
    confusion = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
        batch_size=batch_size,
    )

    accuracy.compute(store=store)
    confusion.compute(store=store)

    assert accuracy.value(store=store) == pytest.approx(0.75)
    np.testing.assert_array_equal(
        confusion.matrix(store=store),
        ((1, 0, 0), (0, 1, 1), (0, 0, 1)),
    )


@pytest.mark.parametrize(
    "value,batched,expected",
    [
        (np.array(("cat",)), False, ("cat",)),
        (np.array((("cat",), ("owl",))), True, ("cat", "owl")),
    ],
)
def test_string_singleton_sparse_shapes_normalize(value, batched, expected):
    assert _normalize_observed_labels(
        value,
        ("cat", "dog", "owl"),
        batched=batched,
        role="true",
    ) == expected


@pytest.mark.parametrize("batch_size", [None, 2])
def test_float_singleton_sparse_shapes_fail_closed(tmp_path, batch_size):
    store = DirStore(tmp_path / str(batch_size))
    true_values = np.array(((0.0,), (1.0,), (0.0,), (1.0,)))
    _model, experiment, test, _trained = _workflow(store, y=true_values)
    confusion = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
        batch_size=batch_size,
    )

    with pytest.raises(ValueError, match="out of range"):
        confusion.compute(store=store)

    assert confusion.compute.results(store=store) == {}


@pytest.mark.parametrize(
    "labels,scores,true_values,match",
    [
        ((0, 1), ((0.9, 0.1),), (2,), "unknown"),
        ((0, 1), ((0.9, 0.1),), ((0, 0, 1),), "out of range"),
        ((0, 1), ((0.9, 0.1),), ((1, 1),), "one-hot"),
        ((0, 1), ((0.5, 0.5),), (0,), "ambiguous"),
    ],
)
def test_confusion_rejects_unknown_out_of_range_and_ambiguous_input_without_publishing(
    tmp_path, labels, scores, true_values, match
):
    store = DirStore(tmp_path / match.replace(" ", "-"))
    _model, experiment, test, _trained = _workflow(
        store,
        scores=scores,
        y=true_values,
    )
    confusion = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=labels,
    )

    with pytest.raises(ValueError, match=match):
        confusion.compute(store=store)

    assert confusion.compute.results(store=store) == {}
    assert confusion.compute.status(store=store).status == "failed"


def test_confusion_rejects_empty_input_without_publishing(tmp_path):
    store = DirStore(tmp_path / "store")
    train = _cache(store, (0,), (0,))
    empty_source = ArrayDataset(
        (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        ),
        spec=(
            TensorSpec("int64", shape=(), backend="numpy"),
            TensorSpec("int64", shape=(), backend="numpy"),
        ),
    )
    empty = CachedDataset(empty_source)
    empty.compute(store=store, representation="numpy-sequence")
    experiment = Experiment(
        TableClassifier(((1.0, 0.0),)),
        NoOpTraining(),
        train_data=train,
    )
    experiment.train(store=store)
    confusion = ConfusionMatrix(
        experiment.train.result,
        empty.compute.result,
        labels=(0, 1),
    )

    with pytest.raises(ValueError, match="empty"):
        confusion.compute(store=store)

    assert confusion.compute.results(store=store) == {}


def test_missing_inputs_fail_without_implicit_model_training_or_cache_compute(tmp_path):
    store = DirStore(tmp_path / "store")
    source = CountingPairs((np.array((0,)), np.array((0,))))
    cached = CachedDataset(source)
    experiment = Experiment(
        TableClassifier(((1.0, 0.0),)),
        NoOpTraining(),
        train_data=cached,
    )
    metric = CategoricalAccuracy(
        experiment.train.result,
        cached.compute.result,
        labels=(0, 1),
    )
    CountingPairs.builds = 0
    NoOpTraining.runs = 0

    with pytest.raises(MissingManagedOutputError, match="active"):
        metric.compute(store=store)

    assert CountingPairs.builds == 0
    assert NoOpTraining.runs == 0
    assert metric.compute.status(store=store).status == "not_started"


def test_reload_reads_metric_without_materializing_model_dataset_or_optional_backends(tmp_path):
    store = DirStore(tmp_path / "store")
    _model, experiment, test, _trained = _workflow(store)
    metric = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )
    metric.compute(store=store)
    Repo(store).save_definition(metric.definition, main=True)
    CountingPairs.builds = 0
    TableClassifier.builds = 0

    reopened_store = DirStore(store.base_dir)
    loaded = Repo(reopened_store).load(metric.definition, restore_state=False)

    np.testing.assert_array_equal(
        loaded.matrix(store=reopened_store),
        ((1, 0, 0), (0, 1, 1), (0, 0, 1)),
    )
    assert CountingPairs.builds == 0
    assert TableClassifier.builds == 0


def test_changed_model_activation_makes_metric_stale_until_rerun_and_failed_rerun_keeps_old_result(
    tmp_path,
):
    store = DirStore(tmp_path / "store")
    model, experiment, test, _trained = _workflow(store)
    metric = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )
    first = metric.compute(store=store)
    expected = metric.matrix(store=store).copy()

    model.scores = np.full((4, 3), 1.0 / 3.0)
    Repo(store).save_object(model, record_policy="none")
    experiment.train.rerun(store=store)

    with pytest.raises(StaleManagedResultError):
        metric.compute(store=store)
    with pytest.raises(ValueError, match="ambiguous"):
        metric.compute.rerun(store=store)

    assert metric.compute.results(store=store)["metric"] == first.outputs["metric"]
    np.testing.assert_array_equal(metric.matrix(store=store), expected)


def test_changed_cache_activation_makes_metric_stale_until_explicit_rerun(tmp_path):
    store = DirStore(tmp_path / "store")
    _model, experiment, test, _trained = _workflow(store)
    metric = CategoricalAccuracy(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )
    first = metric.compute(store=store)

    test.compute.rerun(store=store, representation="numpy-sequence", shard_rows=2)

    with pytest.raises(StaleManagedResultError):
        metric.compute(store=store)
    second = metric.compute.rerun(store=store)

    assert second.realization_id != first.realization_id
    assert metric.value(store=store) == pytest.approx(0.75)
