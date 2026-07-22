"""Record-backed managed classification metric Artifacts."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dryml.artifacts.base import Artifact
from dryml.artifacts.dataset import CachedDataset
from dryml.artifacts.representations import NUMPY_SEQUENCE_KIND, PARQUET_KIND
from dryml.core import Repo
from dryml.core.symbol import resolve_symbol
from dryml.formats.canonical import canonical_json_bytes
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedCapabilityError,
    ManagedMethod,
    ManagedOutput,
    ManagedOutputRef,
    current_operation_context,
    managed,
    resolve_managed_store,
)
from dryml.managed.declarations import resolve_definition_path
from dryml.models.utils import hydrate_model_state
from dryml.records import (
    DataRecord,
    RepresentationSpec,
    StoredStateRecord,
    require_product_integrity,
)

from .scalar import _is_batched, _prediction_pairs, _to_numpy


_RESULT_FILE = "metric.json"
_ACCURACY_SCHEMA = "dryml.metric.categorical_accuracy.v1"
_CONFUSION_SCHEMA = "dryml.metric.confusion_matrix.v1"
_ACCURACY_KIND = "dryml.metric.categorical_accuracy"
_CONFUSION_KIND = "dryml.metric.confusion_matrix"


class _ClassificationMetric(Artifact):
    """Common exact-input contract for managed classification metrics.

    Args:
        model: Logical trained-model output, normally ``experiment.train.result``.
        data: Logical completed-cache output, normally ``cached.compute.result``.
        labels: Unique categorical labels in stable declared order.
        x_path: Path selecting model input from each cache element.
        y_path: Path selecting the true label from each cache element.
        batch_size: Optional positive evaluation batch size.
    """

    _schema: str
    _kind: str

    def __init__(
        self,
        model: ManagedOutputRef,
        data: ManagedOutputRef,
        *,
        labels,
        x_path=0,
        y_path=1,
        batch_size: int | None = None,
    ):
        super().__init__()
        self.model = _require_output_ref(model, "model")
        self.data = _require_output_ref(data, "data")
        self.labels = _normalize_declared_labels(labels)
        self.x_path = _normalize_path(x_path, "x_path")
        self.y_path = _normalize_path(y_path, "y_path")
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise ValueError("batch_size must be a positive integer or None")
        self.batch_size = batch_size
        _model_definition(self.model)
        _require_data_output(self.data)

    def __dryml_managed_inputs__(self, method, args, kwargs):
        if method != "compute" or args or kwargs:
            raise TypeError("classification metric compute accepts only managed runtime arguments")
        return (self.model, self.data)

    def __dryml_managed_validate_inputs__(
        self,
        method,
        args,
        kwargs,
        *,
        store,
        consumed_records,
        consumed_record_links,
    ):
        """Validate exact trained-state and cache records before an attempt starts."""

        if method != "compute" or args or kwargs or consumed_record_links:
            raise TypeError("invalid classification metric input validation request")
        if len(consumed_records) != 2:
            raise ManagedCapabilityError("classification metric requires model and data outputs")
        model_record = StoredStateRecord.from_envelope(
            store.records.read_record(consumed_records[0].record_id)
        )
        expected_model = format_cdef_id(_model_definition(self.model).stable_hash())
        if model_record.subject_cdef_id != expected_model:
            raise ManagedCapabilityError(
                "classification metric model state does not match its declared model"
            )
        data_record = DataRecord.from_envelope(
            store.records.read_record(consumed_records[1].record_id)
        )
        expected_data = format_cdef_id(self.data.producer.stable_hash())
        if data_record.subject_cdef_id != expected_data:
            raise ManagedCapabilityError(
                "classification metric cache does not match its declared dataset"
            )
        representation = RepresentationSpec(
            store.records.read_spec(data_record.representation_id, family="representation")
        )
        if representation.kind not in {NUMPY_SEQUENCE_KIND, PARQUET_KIND}:
            raise ManagedCapabilityError(
                "classification metric cache representation is unsupported"
            )

    @managed(
        outputs=(ManagedOutput("metric", primary=True, kind="data"),),
        resumable=False,
        early_completion=False,
    )
    def compute(self):
        """Evaluate exact active inputs and publish one immutable metric result."""

        context = current_operation_context()
        model = hydrate_model_state(
            _model_definition(self.model),
            context.consumed_records[0].record_id,
            context.store,
        )
        prepare = getattr(model, "prep_eval", None)
        if prepare is not None:
            prepare()
        dataset = Repo(context.store).load_or_build(
            self.data.producer,
            instance="new",
            cache="none",
            restore_state=False,
        )
        if not isinstance(dataset, CachedDataset):
            raise ManagedCapabilityError(
                "classification metric data output must belong to CachedDataset"
            )
        pinned = dataset.view_record(
            context.consumed_records[1].record_id,
            store=context.store,
        )
        payload = self._evaluate(model, pinned)
        encoded = canonical_json_bytes(payload)
        context.write_output(
            "metric",
            _RESULT_FILE,
            (encoded,),
            representation=self._representation(),
            subject_cdef_id=format_cdef_id(self.definition.stable_hash()),
        )

    def result_data(self, repo=None, *, store=None) -> Mapping[str, Any]:
        """Read and validate the active result without materializing either input."""

        selected = resolve_managed_store(repo, store=store, target=self)
        located = self.compute.results(store=selected).get("metric")
        if located is None:
            raise RuntimeError("classification metric has no completed active result")
        envelope = selected.records.read_record(located.record_id)
        record = DataRecord.from_envelope(envelope)
        expected_subject = format_cdef_id(self.definition.stable_hash())
        if any((
            record.subject_cdef_id != expected_subject,
            record.realization_id is None,
            record.output_slot != "metric",
        )):
            raise RuntimeError("classification metric DataRecord ownership is invalid")
        representation = RepresentationSpec(
            selected.records.read_spec(record.representation_id, family="representation")
        )
        if representation.to_envelope() != self._representation().to_envelope():
            raise RuntimeError("classification metric representation metadata is incompatible")
        require_product_integrity(selected.records, envelope)
        if len(record.storage) != 1 or record.storage[0].kind != "product-dir":
            raise RuntimeError("classification metric DataRecord storage is malformed")
        root = selected.records.resolve_storage_ref(
            record.storage[0], record_id=located.record_id
        )
        return self._decode_result(root)

    def _pairs(self, model, data):
        pairs = _prediction_pairs(
            model,
            data,
            x_path=self.x_path,
            y_path=self.y_path,
            batch_size=self.batch_size,
        )
        batched = self.batch_size is not None or _is_batched(data)
        count = 0
        for predicted, true in pairs:
            predicted_labels = _normalize_observed_labels(
                predicted,
                self.labels,
                batched=batched,
                role="predicted",
            )
            true_labels = _normalize_observed_labels(
                true,
                self.labels,
                batched=batched,
                role="true",
            )
            if len(predicted_labels) != len(true_labels):
                raise ValueError(
                    "predicted and true categorical labels have incompatible batch sizes"
                )
            for predicted_label, true_label in zip(predicted_labels, true_labels):
                count += 1
                yield predicted_label, true_label
        if count == 0:
            raise ValueError("cannot compute a classification metric on empty input")

    def _representation(self) -> RepresentationSpec:
        parameters = {"labels": self.labels}
        if self._kind == _CONFUSION_KIND:
            parameters.update({"rows": "true", "columns": "predicted"})
        return RepresentationSpec.create(
            self._kind,
            version="1",
            parameters=parameters,
            traits=("classification-metric", "json", "stream-readable"),
            storage_kinds=("product-dir",),
            payload={"file": _RESULT_FILE, "schema": self._schema},
        )

    def _decode_result(self, root: Path) -> Mapping[str, Any]:
        try:
            value = json.loads((root / _RESULT_FILE).read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("classification metric result is unreadable") from exc
        if not isinstance(value, dict):
            raise RuntimeError("classification metric result must be a JSON object")
        return value

    def _evaluate(self, model, data) -> dict[str, Any]:
        raise NotImplementedError


class CategoricalAccuracy(_ClassificationMetric):
    """Managed categorical accuracy over exact trained-model/cache outputs."""

    _schema = _ACCURACY_SCHEMA
    _kind = _ACCURACY_KIND

    def _evaluate(self, model, data):
        correct = 0
        count = 0
        for predicted, true in self._pairs(model, data):
            count += 1
            correct += predicted == true
        return {
            "schema": self._schema,
            "schema_version": 1,
            "metric": "categorical_accuracy",
            "labels": list(self.labels),
            "correct": correct,
            "count": count,
            "value": correct / count,
        }

    def value(self, repo=None, *, store=None) -> float:
        """Return the validated scalar value from the active realization."""

        value = self.result_data(repo=repo, store=store)
        expected = {
            "schema",
            "schema_version",
            "metric",
            "labels",
            "correct",
            "count",
            "value",
        }
        if set(value) != expected or value.get("schema") != self._schema:
            raise RuntimeError("categorical accuracy result fields are malformed")
        if value.get("schema_version") != 1 or value.get("metric") != "categorical_accuracy":
            raise RuntimeError("categorical accuracy result schema is incompatible")
        if tuple(value.get("labels", ())) != self.labels:
            raise RuntimeError("categorical accuracy result label order is incompatible")
        count = value.get("count")
        correct = value.get("correct")
        scalar = value.get("value")
        if not _valid_accuracy_values(count, correct, scalar):
            raise RuntimeError("categorical accuracy result values are malformed")
        return float(scalar)


class ConfusionMatrix(_ClassificationMetric):
    """Managed confusion matrix with true rows and predicted columns."""

    _schema = _CONFUSION_SCHEMA
    _kind = _CONFUSION_KIND

    def _evaluate(self, model, data):
        index = {label: position for position, label in enumerate(self.labels)}
        matrix = np.zeros((len(self.labels), len(self.labels)), dtype=np.int64)
        count = 0
        for predicted, true in self._pairs(model, data):
            matrix[index[true], index[predicted]] += 1
            count += 1
        return {
            "schema": self._schema,
            "schema_version": 1,
            "metric": "confusion_matrix",
            "labels": list(self.labels),
            "rows": "true",
            "columns": "predicted",
            "count": count,
            "matrix": matrix.tolist(),
        }

    def matrix(self, repo=None, *, store=None) -> np.ndarray:
        """Return a validated matrix whose rows are true and columns predicted."""

        value = self.result_data(repo=repo, store=store)
        expected = {
            "schema",
            "schema_version",
            "metric",
            "labels",
            "rows",
            "columns",
            "count",
            "matrix",
        }
        if set(value) != expected or value.get("schema") != self._schema:
            raise RuntimeError("confusion matrix result fields are malformed")
        if any((
            value.get("schema_version") != 1,
            value.get("metric") != "confusion_matrix",
            value.get("rows") != "true",
            value.get("columns") != "predicted",
            tuple(value.get("labels", ())) != self.labels,
        )):
            raise RuntimeError("confusion matrix result metadata is incompatible")
        matrix = np.asarray(value.get("matrix"))
        count = value.get("count")
        if not _valid_matrix_values(matrix, count, len(self.labels)):
            raise RuntimeError("confusion matrix result values are malformed")
        return matrix.astype(np.int64, copy=False)


def _require_output_ref(value, name: str) -> ManagedOutputRef:
    if not isinstance(value, ManagedOutputRef):
        raise TypeError(f"classification metric {name} must be a ManagedOutputRef")
    return value


def _output_declaration(ref: ManagedOutputRef):
    cls = resolve_symbol(ref.producer.cls)
    try:
        descriptor = inspect.getattr_static(cls, ref.method)
    except AttributeError as exc:
        raise TypeError("classification metric input has no declared managed method") from exc
    if not isinstance(descriptor, ManagedMethod):
        raise TypeError("classification metric input method is not managed")
    declaration = descriptor.output_declarations(ref.producer).get(ref.slot)
    if declaration is None:
        raise TypeError("classification metric input references an unknown output slot")
    return declaration


def _model_definition(ref: ManagedOutputRef):
    declaration = _output_declaration(ref)
    if declaration.kind not in {"object", "stored_state"} or declaration.subject_path is None:
        raise TypeError("classification metric model must reference declared stored model state")
    return resolve_definition_path(ref.producer, declaration.subject_path)


def _require_data_output(ref: ManagedOutputRef) -> None:
    declaration = _output_declaration(ref)
    cls = resolve_symbol(ref.producer.cls)
    if declaration.kind != "data" or not issubclass(cls, CachedDataset):
        raise TypeError("classification metric data must reference a CachedDataset output")


def _normalize_declared_labels(labels) -> tuple[int | str, ...]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, (tuple, list)):
        raise TypeError("labels must be a declared tuple or list")
    result = []
    for label in labels:
        if isinstance(label, np.generic):
            label = label.item()
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise TypeError("labels must contain only integer or string categories")
        result.append(label)
    if len(result) < 2:
        raise ValueError("labels must declare at least two categories")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique")
    return tuple(result)


def _normalize_path(value, name: str):
    path = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not path or any(not isinstance(part, (str, int)) for part in path):
        raise TypeError(f"{name} must contain string or integer path segments")
    return path


def _normalize_observed_labels(value, labels, *, batched: bool, role: str):
    array = _to_numpy(value)
    if array.dtype.hasobject:
        raise ValueError(f"{role} categorical labels are ambiguous object values")
    if batched:
        if array.ndim == 1:
            return tuple(_sparse_label(item, labels, role) for item in array)
        if array.ndim == 2:
            if array.shape[1] == 1 and array.dtype.kind in "iuU":
                return tuple(_sparse_label(row[0], labels, role) for row in array)
            return tuple(_vector_label(row, labels, role) for row in array)
        raise ValueError(f"{role} categorical labels have ambiguous batched shape {array.shape}")
    if array.ndim == 0:
        return (_sparse_label(array.item(), labels, role),)
    if array.ndim == 1:
        if array.size == 1 and array.dtype.kind in "iuU":
            return (_sparse_label(array[0], labels, role),)
        return (_vector_label(array, labels, role),)
    raise ValueError(f"{role} categorical label has ambiguous shape {array.shape}")


def _sparse_label(value, labels, role: str):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{role} categorical label is ambiguous")
    if value not in labels:
        raise ValueError(f"{role} categorical label {value!r} is unknown")
    return value


def _vector_label(value, labels, role: str):
    vector = np.asarray(value)
    if vector.ndim != 1:
        raise ValueError(f"{role} categorical vector is ambiguous")
    if vector.size != len(labels):
        raise ValueError(
            f"{role} categorical vector width {vector.size} is out of range for {len(labels)} labels"
        )
    if vector.dtype.kind not in "biuf" or not np.all(np.isfinite(vector)):
        raise ValueError(f"{role} categorical vector is ambiguous")
    if role == "true":
        if not np.all((vector == 0) | (vector == 1)) or int(np.sum(vector)) != 1:
            raise ValueError("true categorical vector is ambiguous; expected one-hot labels")
    maximum = np.max(vector)
    winners = np.flatnonzero(vector == maximum)
    if len(winners) != 1:
        raise ValueError(f"{role} categorical vector is ambiguous")
    return labels[int(winners[0])]


def _valid_accuracy_values(count, correct, scalar) -> bool:
    if type(count) is not int or count < 1:
        return False
    if type(correct) is not int or not 0 <= correct <= count:
        return False
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        return False
    return np.isfinite(scalar) and float(scalar) == correct / count


def _valid_matrix_values(matrix, count, label_count: int) -> bool:
    if matrix.shape != (label_count, label_count) or matrix.dtype.kind not in "iu":
        return False
    if type(count) is not int or count < 1:
        return False
    return not np.any(matrix < 0) and int(matrix.sum()) == count


CategoricalAccuracy.__module__ = "dryml.metrics"
ConfusionMatrix.__module__ = "dryml.metrics"


__all__ = ["CategoricalAccuracy", "ConfusionMatrix"]
