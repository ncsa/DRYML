"""Managed, record-backed cached Dataset realizations."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from dryml.artifacts.base import Artifact
from dryml.artifacts.representations.numpy_sequence import (
    NUMPY_SEQUENCE_KIND,
    NUMPY_SEQUENCE_REPRESENTATION,
    NumpySequenceIndex,
    iter_numpy_sequence,
    write_numpy_sequence_stream,
)
from dryml.artifacts.representations.parquet import (
    PARQUET_KIND,
    PARQUET_REPRESENTATION,
    numpy_to_parquet_adapter_registry,
)
from dryml.core2 import RefCDef, Repo
from dryml.core2.cardinality import Cardinality
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.tensor_spec import SpecTree
from dryml.data.dataset import Dataset, as_cardinality, dataset_cardinality
from dryml.data.resume import (
    ResumeMode,
    dataset_definition_metadata,
    dataset_resume_capability,
    open_resumable_dataset,
)
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ControlRequest,
    ManagedCapabilityError,
    ManagedOutput,
    OperationPreflight,
    current_operation_context,
    managed,
)
from dryml.records import (
    AdapterExecutionResult,
    AdapterSearchLimits,
    DataRecord,
    LocatedTypedRecord,
    RecordResolutionIssue,
    RepresentationRequirement,
    RecordValidationError,
    RepresentationSpec,
    resolve_data_record,
    run_adapter_plan,
    require_checkpoint_integrity,
    require_product_integrity,
)


_CHECKPOINT_SCHEMA = "dryml.cached-dataset.v1"
_CHECKPOINT_FILE = "cached-dataset.pkl"
_DEFAULT_SHARD_ROWS = 1024
_DEFAULT_SHARD_BYTES = 64 * 1024 * 1024


class CachedDataset(Artifact, Dataset):
    """A lightweight Dataset definition backed by one active managed result.

    ``src`` is a non-materializing :class:`~dryml.core2.RefCDef` edge. Element
    specification, cardinality, and order semantics are copied into this
    definition so loading or iterating a completed cache never constructs the
    source. Heavy rows live only in immutable sharded ``DataRecord`` products.

    Args:
        src: Source Dataset Object or exact source definition.
        spec: Lightweight element metadata. Inferred from a live source.
        cardinality: Lightweight source cardinality. Inferred when available.
        order: Stable description of yielded order; currently ``"source"``.
    """

    @classmethod
    def __prepare_args__(
        cls,
        src,
        *,
        spec: SpecTree | None = None,
        cardinality: Cardinality | int | None = None,
        order: str = "source",
    ):
        args, kwargs = super().__prepare_args__(
            src,
            spec=spec,
            cardinality=cardinality,
            order=order,
        )
        source = args[0]
        if isinstance(source, Dataset):
            if kwargs["spec"] is None:
                kwargs["spec"] = source.spec
            if kwargs["cardinality"] is None:
                kwargs["cardinality"] = dataset_cardinality(source)
        elif isinstance(source, ConcreteDefinition) and (
            kwargs["spec"] is None or kwargs["cardinality"] is None
        ):
            source_spec, source_cardinality = dataset_definition_metadata(source)
            if kwargs["spec"] is None:
                kwargs["spec"] = source_spec
            if kwargs["cardinality"] is None:
                kwargs["cardinality"] = source_cardinality
        if kwargs["spec"] is None:
            raise ValueError("CachedDataset requires element spec metadata")
        if kwargs["cardinality"] is None:
            kwargs["cardinality"] = Cardinality.UNKNOWN
        kwargs["cardinality"] = as_cardinality(kwargs["cardinality"])
        if kwargs["cardinality"].is_infinite:
            raise ValueError("CachedDataset cannot materialize an infinite source")
        if kwargs["order"] != "source":
            raise ValueError("CachedDataset currently supports only source-defined order")
        return (source,), kwargs

    def __init__(
        self,
        src: RefCDef,
        *,
        spec: SpecTree,
        cardinality: Cardinality,
        order: str = "source",
    ):
        if not isinstance(src, ConcreteDefinition):
            raise TypeError("CachedDataset src must resolve to a ConcreteDefinition")
        self.src = src
        self.cardinality = as_cardinality(cardinality)
        self.order = order
        Dataset.__init__(self, spec=spec)

    def __len__(self) -> Cardinality:
        """Return definition metadata without resolving a realization."""

        return self.cardinality

    def __iter__(self) -> Iterator[Any]:
        """Iterate the active completed cache in the default repository scope."""

        return iter(self.view())

    def view(self, repo=None, *, store=None) -> "CachedDatasetView":
        """Return an iterable pinned to one explicit repository authority."""

        return CachedDatasetView(self, repo=repo, store=store)

    def active_record(self, repo=None, *, store=None):
        """Return the exact validated active ``DataRecord`` and product root.

        This is the lightweight prerequisite seam used by later managed
        consumers such as training. It never computes or materializes ``src``.
        """

        located, record, root, representation, _selected = self._active_output_record(
            repo=repo, store=store
        )
        if representation.kind != NUMPY_SEQUENCE_KIND:
            raise RuntimeError("active cache has no supported NumPy sequence representation")
        return located, record, root

    def request_representation(
        self,
        representation,
        *,
        repo=None,
        store=None,
        adapters=None,
        limits: AdapterSearchLimits | None = None,
    ) -> AdapterExecutionResult:
        """Reuse or derive one representation of the exact active realization.

        The request never invokes ``compute`` and never changes active
        realization selection. Adapter failures are returned structurally.
        """

        requirement = _representation_requirement(representation)
        located, record, _root, _spec, selected = self._active_output_record(
            repo=repo, store=store
        )
        selected_repo = Repo(selected)
        registry = adapters if adapters is not None else numpy_to_parquet_adapter_registry()
        resolution = resolve_data_record(
            selected_repo,
            LocatedTypedRecord(located, record),
            requirement,
            adapters=registry,
            limits=limits,
        )
        if resolution.status == "ok" and resolution.selected is not None:
            try:
                require_product_integrity(
                    selected.records,
                    selected.records.read_record(resolution.selected.ref.record_id),
                )
            except Exception as exc:
                return AdapterExecutionResult(
                    "failed",
                    issues=(RecordResolutionIssue("product_integrity_failed", str(exc)),),
                )
            return AdapterExecutionResult("ok", target_records=(resolution.selected.ref,))
        if resolution.status != "requires_adapter":
            return AdapterExecutionResult(
                resolution.status if resolution.status in {"not_found", "unsupported", "failed"} else "failed",
                issues=resolution.report.issues,
            )
        return run_adapter_plan(
            resolution.adapter_plan,
            repo=selected_repo,
            store=selected,
            registry=registry,
        )

    def representation_record(self, representation, *, repo=None, store=None, adapters=None, limits=None):
        """Return one validated representation record and product root."""

        result = self.request_representation(
            representation,
            repo=repo,
            store=store,
            adapters=adapters,
            limits=limits,
        )
        if result.status != "ok" or not result.target_records:
            message = result.issues[0].message if result.issues else "representation request failed"
            raise RuntimeError(message)
        from dryml.managed.store import resolve_managed_store

        selected = resolve_managed_store(repo, store=store, target=self)
        located = result.target_records[-1]
        envelope = selected.records.read_record(located.record_id)
        record = DataRecord.from_envelope(envelope)
        require_product_integrity(selected.records, envelope)
        if len(record.storage) != 1 or record.storage[0].kind != "product-dir":
            raise RecordValidationError("cache representation DataRecord storage is malformed")
        root = selected.records.resolve_storage_ref(record.storage[0], record_id=located.record_id)
        spec = RepresentationSpec(
            selected.records.read_spec(record.representation_id, family="representation")
        )
        return located, record, root, spec

    def tensorflow_view(self, repo=None, *, store=None, representation="numpy-sequence"):
        """Return a dependency-lazy TensorFlow iterable over a resolved cache."""

        from dryml.data.tf.cache import TensorFlowCacheView

        return TensorFlowCacheView(self, repo=repo, store=store, representation=representation)

    def torch_view(self, repo=None, *, store=None, representation="numpy-sequence"):
        """Return a dependency-lazy PyTorch iterable over a resolved cache."""

        from dryml.data.torch.cache import TorchCacheView

        return TorchCacheView(self, repo=repo, store=store, representation=representation)

    def _active_output_record(self, repo=None, *, store=None):
        results = self.compute.results(repo=repo, store=store)
        located = results.get("data")
        if located is None:
            raise RuntimeError("CachedDataset has no completed compatible active realization")
        from dryml.managed.store import resolve_managed_store

        selected = resolve_managed_store(repo, store=store, target=self)
        envelope = selected.records.read_record(located.record_id)
        record = DataRecord.from_envelope(envelope)
        if record.output_slot != "data" or record.realization_id is None:
            raise RecordValidationError("active cache DataRecord has invalid managed ownership")
        representation = RepresentationSpec(selected.records.read_spec(
            record.representation_id, family="representation"
        ))
        require_product_integrity(selected.records, envelope)
        if len(record.storage) != 1 or record.storage[0].kind != "product-dir":
            raise RecordValidationError("active cache DataRecord storage is malformed")
        root = selected.records.resolve_storage_ref(
            record.storage[0],
            record_id=located.record_id,
        )
        return located, record, root, representation, selected

    def __dryml_managed_preflight__(self, method, args, kwargs):
        if method != "compute":
            raise ManagedCapabilityError(f"unsupported CachedDataset managed method {method!r}")
        representation = _invocation_representation(args, kwargs)
        if representation is not None:
            _normalize_representation(representation)
        _optional_limit(kwargs.get("shard_rows"), "shard_rows")
        _optional_limit(kwargs.get("shard_bytes"), "shard_bytes")
        capability = dataset_resume_capability(self.src)
        return OperationPreflight(
            resumable=capability.mode is ResumeMode.EXACT,
            checkpoint_schema=_CHECKPOINT_SCHEMA if capability.mode is ResumeMode.EXACT else None,
            early_completion=False,
        )

    def __dryml_managed_validate_invocation__(
        self,
        method,
        args,
        kwargs,
        *,
        store,
        operation,
        has_active,
        has_pending,
        rerun,
    ):
        del method, store
        representation = _invocation_representation(args, kwargs)
        if representation is None and (rerun or not (has_active or has_pending)):
            raise ManagedCapabilityError(
                "the first CachedDataset realization must explicitly choose representation='numpy-sequence'"
            )
        if has_pending and not rerun:
            checkpoint = _pending_checkpoint(operation, self)
            if checkpoint is not None:
                requested_id = (
                    None
                    if representation is None
                    else _normalize_representation(representation).id
                )
                if requested_id is not None and requested_id != checkpoint["representation_id"]:
                    raise ManagedCapabilityError(
                        "resume representation does not match the checkpoint"
                    )
                for name in ("shard_rows", "shard_bytes"):
                    value = kwargs.get(name)
                    if value is not None and value != checkpoint[name]:
                        raise ManagedCapabilityError(
                            f"resume {name} does not match the checkpoint"
                        )

    @managed(
        outputs=(
            ManagedOutput(
                "data",
                primary=True,
                kind="data",
                representations=(NUMPY_SEQUENCE_KIND,),
            ),
        ),
        resumable=True,
        checkpoint_schema=_CHECKPOINT_SCHEMA,
        early_completion=False,
    )
    def compute(
        self,
        representation=None,
        *,
        shard_rows: int | None = None,
        shard_bytes: int | None = None,
    ):
        """Stream the source into bounded NumPy shards under managed control.

        The first realization requires an explicit ``"numpy-sequence"``
        representation. Exact pipelines persist the complete source/stage
        cursor after each durable shard; replay-only and unsupported pipelines
        may compute normally but require explicit rerun after interruption.
        """

        context = current_operation_context()
        checkpoint = _load_checkpoint(context) if context.is_resume else None
        if checkpoint is not None:
            representation_spec = _normalize_representation(
                representation or checkpoint["representation_id"]
            )
            rows = checkpoint["shard_rows"] if shard_rows is None else shard_rows
            size = checkpoint["shard_bytes"] if shard_bytes is None else shard_bytes
            if representation_spec.id != checkpoint["representation_id"]:
                raise ManagedCapabilityError("resume representation does not match the checkpoint")
            if rows != checkpoint["shard_rows"] or size != checkpoint["shard_bytes"]:
                raise ManagedCapabilityError("resume shard configuration does not match the checkpoint")
            prior = NumpySequenceIndex.from_json(checkpoint["index"])
            pipeline_state = checkpoint["pipeline"]
            checkpoint_serial = checkpoint["checkpoint_serial"]
            _restore_prior_shards(context, prior)
        else:
            representation_spec = _normalize_representation(representation)
            rows = _DEFAULT_SHARD_ROWS if shard_rows is None else shard_rows
            size = _DEFAULT_SHARD_BYTES if shard_bytes is None else shard_bytes
            prior = NumpySequenceIndex(0, None, ())
            pipeline_state = None
            checkpoint_serial = 0
        _required_limit(rows, "shard_rows")
        _required_limit(size, "shard_bytes")

        source_repo = Repo(context.store)
        source = source_repo.load_or_build(self.src)
        capability = dataset_resume_capability(self.src)
        exact = capability.mode is ResumeMode.EXACT
        iterator = (
            open_resumable_dataset(source, pipeline_state)
            if exact
            else iter(source)
        )
        latest_index = prior

        def durable_checkpoint() -> None:
            nonlocal checkpoint_serial
            checkpoint_serial += 1
            _commit_checkpoint(
                context,
                source_cdef_id=format_cdef_id(self.src.stable_hash()),
                representation_id=representation_spec.id,
                shard_rows=rows,
                shard_bytes=size,
                checkpoint_serial=checkpoint_serial,
                index=latest_index,
                pipeline=iterator.checkpoint(),
            )

        def write_file(path: str, payload: bytes) -> None:
            context.write_output(
                "data",
                path,
                (payload,),
                representation=representation_spec,
                subject_cdef_id=format_cdef_id(self.definition.stable_hash()),
            )

        def commit(index: NumpySequenceIndex) -> None:
            nonlocal latest_index
            latest_index = index
            total = self.cardinality.value if self.cardinality.is_finite else None
            context.progress(index.count, total=total, message="cached dataset rows")
            control = context.safe_point(
                checkpoint=durable_checkpoint if exact else None
            )
            if control is ControlRequest.GRACEFUL_STOP:
                raise ManagedCapabilityError("CachedDataset cannot complete from a source prefix")
            if exact:
                durable_checkpoint()

        final_index = write_numpy_sequence_stream(
            iterator,
            write_file,
            shard_rows=rows,
            shard_bytes=size,
            prior=prior,
            on_flush=commit,
        )
        if exact and final_index.count == prior.count:
            commit(final_index)
        if self.cardinality.is_finite and final_index.count != self.cardinality.require_finite():
            raise RuntimeError(
                "CachedDataset source cardinality changed while materializing "
                f"({final_index.count} rows, expected {self.cardinality.require_finite()})"
            )


@dataclass(frozen=True, slots=True)
class CachedDatasetView:
    """Dataset-compatible iterable pinned to one cache authority."""

    dataset: CachedDataset
    repo: Any = None
    store: Any = None

    @property
    def spec(self):
        """Return element metadata from the lightweight definition."""

        return self.dataset.spec

    def __len__(self):
        """Return lightweight cardinality metadata."""

        return self.dataset.__len__()

    def __iter__(self):
        """Validate and lazily read only the selected active DataRecord."""

        _located, _record, root = self.dataset.active_record(
            repo=self.repo,
            store=self.store,
        )
        yield from iter_numpy_sequence(root)


def _normalize_representation(value):
    if isinstance(value, str) and value in {
        "numpy",
        "numpy-sequence",
        NUMPY_SEQUENCE_KIND,
        NUMPY_SEQUENCE_REPRESENTATION.id,
    }:
        return NUMPY_SEQUENCE_REPRESENTATION
    if isinstance(value, RepresentationSpec) and value.kind == NUMPY_SEQUENCE_KIND:
        return value
    raise ManagedCapabilityError(
        "CachedDataset currently supports only the explicit 'numpy-sequence' representation"
    )


def _representation_requirement(value):
    if isinstance(value, RepresentationRequirement):
        return value
    if isinstance(value, RepresentationSpec):
        if value.kind not in {NUMPY_SEQUENCE_KIND, PARQUET_KIND}:
            raise ManagedCapabilityError("CachedDataset representation is unsupported")
        return RepresentationRequirement(kind=value.kind, representation_id=value.id)
    if isinstance(value, str):
        if value in {"numpy", "numpy-sequence", NUMPY_SEQUENCE_KIND, NUMPY_SEQUENCE_REPRESENTATION.id}:
            return RepresentationRequirement(
                kind=NUMPY_SEQUENCE_KIND,
                representation_id=NUMPY_SEQUENCE_REPRESENTATION.id,
            )
        if value in {"parquet", PARQUET_KIND, PARQUET_REPRESENTATION.id}:
            return RepresentationRequirement(
                kind=PARQUET_KIND,
                representation_id=PARQUET_REPRESENTATION.id,
            )
    raise ManagedCapabilityError("CachedDataset representation is unsupported")


def _invocation_representation(args, kwargs):
    if len(args) > 1:
        raise TypeError("CachedDataset.compute accepts at most one positional argument")
    if args:
        if "representation" in kwargs:
            raise TypeError("compute() got multiple values for representation")
        return args[0]
    return kwargs.get("representation")


def _optional_limit(value, name: str) -> None:
    if value is not None:
        _required_limit(value, name)


def _required_limit(value, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ManagedCapabilityError(f"{name} must be a positive integer")


def _commit_checkpoint(
    context,
    *,
    source_cdef_id,
    representation_id,
    shard_rows,
    shard_bytes,
    checkpoint_serial,
    index,
    pipeline,
):
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "source_cdef_id": source_cdef_id,
        "representation_id": representation_id,
        "shard_rows": shard_rows,
        "shard_bytes": shard_bytes,
        "checkpoint_serial": checkpoint_serial,
        "index": index.to_json(),
        "pipeline": pipeline,
    }
    context.write_checkpoint(_CHECKPOINT_FILE, (pickle.dumps(payload, protocol=5),))
    context.commit_checkpoint(
        metadata={
            "rows": index.count,
            "shards": len(index.shards),
            "serial": checkpoint_serial,
        }
    )


def _load_checkpoint(context) -> Mapping[str, Any]:
    root = context.checkpoint_path
    if root is None:
        raise RuntimeError("resumed CachedDataset has no committed checkpoint")
    return _decode_checkpoint(root, context.producer)


def _decode_checkpoint(root, producer) -> Mapping[str, Any]:
    try:
        payload = pickle.loads((root / _CHECKPOINT_FILE).read_bytes())
    except Exception as exc:
        raise RuntimeError("CachedDataset checkpoint payload is unreadable") from exc
    fields = {
        "schema",
        "schema_version",
        "source_cdef_id",
        "representation_id",
        "shard_rows",
        "shard_bytes",
        "checkpoint_serial",
        "index",
        "pipeline",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise RuntimeError("CachedDataset checkpoint payload is malformed")
    if payload.get("schema") != _CHECKPOINT_SCHEMA or payload.get("schema_version") != 1:
        raise RuntimeError("CachedDataset checkpoint schema is incompatible")
    if payload.get("source_cdef_id") != format_cdef_id(producer.src.stable_hash()):
        raise RuntimeError("CachedDataset checkpoint source is incompatible")
    if type(payload.get("checkpoint_serial")) is not int or payload["checkpoint_serial"] < 1:
        raise RuntimeError("CachedDataset checkpoint serial is malformed")
    return payload


def _pending_checkpoint(operation, producer) -> Mapping[str, Any] | None:
    control = operation._read_control(missing_ok=True)
    if control is None or control.pending_realization_id is None:
        return None
    state = operation._read_realization(control.pending_realization_id)
    if state.checkpoint_head is None:
        return None
    roots = tuple(
        operation.attempts_dir.glob(f"*/checkpoints/{state.checkpoint_head}")
    )
    if not roots:
        raise RuntimeError("CachedDataset committed checkpoint payload is missing")
    for root in roots:
        require_checkpoint_integrity(root, state.checkpoint_head)
    payloads = tuple((root / _CHECKPOINT_FILE).read_bytes() for root in roots)
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise RuntimeError("CachedDataset checkpoint identity resolves to conflicting payloads")
    return _decode_checkpoint(roots[0], producer)


def _restore_prior_shards(context, index: NumpySequenceIndex) -> None:
    for shard in index.shards:
        path = context.writer.retain_output_file("data", shard.path)
        if path.stat().st_size != shard.size:
            raise RuntimeError("retained CachedDataset shard size does not match checkpoint")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != shard.sha256:
            raise RuntimeError("retained CachedDataset shard digest does not match checkpoint")


CachedDataset.__module__ = "dryml.artifacts"


__all__ = ["CachedDataset", "CachedDatasetView"]
