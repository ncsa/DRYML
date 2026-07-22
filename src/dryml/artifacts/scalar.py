from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

from dryml.core import RefCDef, Repo
from dryml.core.repo import get_default_repo
from dryml.core.utils.recurse import iter_leaves
from dryml.formats.canonical import canonical_json_bytes
from dryml.formats.refs import format_cdef_id
from dryml.managed import ManagedOutput, current_operation_context, managed, resolve_managed_store
from dryml.records import DataRecord, RepresentationSpec, require_product_integrity

from .base import Artifact


_SCALAR_FILENAME = "value.json"
_SCALAR_REPRESENTATION = RepresentationSpec.create(
    "dryml.scalar",
    version="1",
    traits=("scalar", "json", "stream-readable"),
    storage_kinds=("product-dir",),
    payload={"file": _SCALAR_FILENAME, "schema": "dryml.scalar.v1"},
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class Scalar(Artifact):
    """Lightweight immediate scalar whose value is definition data."""

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def compute(self, repo=None, *, store=None):
        """Return the constructor value without creating a realization."""

        return self.value


class ScalarAgg(Artifact):
    """Base for scalar reductions published through managed ``DataRecord`` output."""

    def __init__(self, src: RefCDef):
        super().__init__()
        self.src = src

    def aggregate(self, values: Iterable[Any]):
        raise NotImplementedError

    @managed(outputs=(ManagedOutput("value", primary=True, kind="data"),))
    def compute(self):
        """Reduce the source directly or publish a managed scalar realization."""

        try:
            context = current_operation_context()
        except RuntimeError:
            source = (get_default_repo() or Repo()).load_or_build(self.src)
            value = _json_scalar(self.aggregate(iter(source)))
            self.value = value
            return value

        source = Repo(context.store).load_or_build(
            self.src,
            instance="new",
            cache="none",
            restore_state=False,
        )
        value = _json_scalar(self.aggregate(iter(source)))
        payload = canonical_json_bytes(
            {"schema": "dryml.scalar.v1", "schema_version": 1, "value": value}
        )
        context.write_output(
            "value",
            _SCALAR_FILENAME,
            (payload,),
            representation=_SCALAR_REPRESENTATION,
            subject_cdef_id=format_cdef_id(self.definition.stable_hash()),
        )

    def read(self, repo=None, *, store=None):
        """Return the validated scalar from the active managed realization."""

        selected = resolve_managed_store(repo, store=store, target=self)
        located = self.compute.results(store=selected).get("value")
        if located is None:
            raise RuntimeError("scalar aggregate has no completed active result")
        envelope = selected.records.read_record(located.record_id)
        record = DataRecord.from_envelope(envelope)
        if any((
            record.subject_cdef_id != format_cdef_id(self.definition.stable_hash()),
            record.realization_id is None,
            record.output_slot != "value",
            record.representation_id != _SCALAR_REPRESENTATION.id,
        )):
            raise RuntimeError("scalar aggregate DataRecord is incompatible")
        require_product_integrity(selected.records, envelope)
        if len(record.storage) != 1 or record.storage[0].kind != "product-dir":
            raise RuntimeError("scalar aggregate DataRecord storage is malformed")
        root = selected.records.resolve_storage_ref(
            record.storage[0], record_id=located.record_id
        )
        try:
            payload = json.loads((root / _SCALAR_FILENAME).read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("scalar aggregate result is unreadable") from exc
        if not _valid_scalar_payload(payload):
            raise RuntimeError("scalar aggregate result is malformed")
        return _json_scalar(payload["value"])


class ScalarAvg(ScalarAgg):
    def aggregate(self, values: Iterable[Any]) -> float:
        total = 0.0
        count = 0
        for item in values:
            for leaf in iter_leaves(item):
                arr = _as_numpy(leaf)
                if arr.size == 0:
                    continue
                total += float(np.sum(arr))
                count += int(arr.size)

        if count == 0:
            raise ValueError("Cannot average an empty scalar source.")
        return total / count


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("scalar aggregate results must be integer or floating-point values")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("scalar aggregate results must be finite")
    return value


def _valid_scalar_payload(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    return all((
        set(payload) == {"schema", "schema_version", "value"},
        payload.get("schema") == "dryml.scalar.v1",
        payload.get("schema_version") == 1,
    ))


Scalar.__module__ = "dryml.artifacts"
ScalarAgg.__module__ = "dryml.artifacts"
ScalarAvg.__module__ = "dryml.artifacts"
