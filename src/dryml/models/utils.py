from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dryml.core import Repo
from dryml.core.definition import ConcreteDefinition
from dryml.core.tensor_spec import iter_specs
from dryml.data import Shuffle, Take, Unbatch
from dryml.formats.refs import format_cdef_id
from dryml.models.train_spec import TrainState
from dryml.records import (
    LocatedRecordRef,
    ProductManifest,
    ProductManifestEntry,
    ProductWriteSession,
    RecordPolicyOptions,
    StorageRef,
    StoredStateRecord,
    attach_record_id,
    default_object_state_representation_spec,
    require_product_integrity,
)


MODEL_STATE_REPRESENTATION = default_object_state_representation_spec(
    RecordPolicyOptions(
        representation_payload={
            "storage_kind": "product-dir",
            "role": "model-state",
            "description": "Immutable managed DRYML model state.",
        }
    )
)


def validate_num_examples(num_examples: int | None) -> None:
    if num_examples is not None and num_examples < 0:
        raise ValueError("num_examples must be non-negative or None.")


def dataset_is_batched(dataset) -> bool:
    try:
        return any(spec.batched for spec in iter_specs(dataset.spec))
    except ValueError:
        return False


def finite_dataset_len(dataset) -> int | None:
    try:
        cardinality = dataset.__len__()
    except Exception:
        return None

    if hasattr(cardinality, "is_finite"):
        if cardinality.is_finite:
            return cardinality.require_finite()
        return None
    return int(cardinality)


def prepare_training_data(
    train_data,
    *,
    num_examples: int | None = None,
    shuffle: bool = False,
    shuffle_seed=None,
    shuffle_buffer_size: int | None = None,
):
    if train_data is None:
        raise ValueError("Experiment has no train_data.")
    validate_num_examples(num_examples)

    if dataset_is_batched(train_data):
        train_data = Unbatch(train_data)

    if shuffle:
        buffer_size = shuffle_buffer_size or finite_dataset_len(train_data)
        if buffer_size is None:
            raise ValueError("shuffle_buffer_size is required when train_data length is unknown.")
        train_data = Shuffle(train_data, buffer_size, seed=shuffle_seed)

    if num_examples is not None:
        train_data = Take(train_data, num_examples)

    return train_data


def advance_train_state(exp, *, epochs: int = 0, steps: int = 0, phase: str = TrainState.trained):
    if epochs:
        exp.state.advance_epoch(epochs)
    if steps:
        exp.state.advance_step(steps)
    exp.state.phase = phase


def snapshot_model_state(model: ConcreteDefinition, store) -> LocatedRecordRef:
    """Copy selected ordinary model state into one immutable product record.

    The source directory is collected before and after the bounded copy.
    Detected changes retry and then fail instead of publishing mixed bytes.
    """

    if not isinstance(model, ConcreteDefinition):
        raise TypeError("model state snapshots require an exact ConcreteDefinition")
    store.records.write_spec(MODEL_STATE_REPRESENTATION, family="representation")
    source = Path(store.object_dir(model))
    if _has_state_files(source):
        return _snapshot_state_tree(model, source, store)

    fresh = Repo(store).load_or_build(
        model,
        instance="new",
        cache="none",
        restore_state=False,
    )
    with tempfile.TemporaryDirectory(prefix="dryml-initial-model-") as temp:
        fresh.save_state_to_dir(temp)
        return _snapshot_state_tree(model, Path(temp), store)


def hydrate_model_state(model: ConcreteDefinition, record_id: str, store):
    """Restore one product-backed state record into a new uncached model."""

    if not isinstance(model, ConcreteDefinition):
        raise TypeError("model state hydration requires an exact ConcreteDefinition")
    envelope = store.records.read_record(record_id)
    record = StoredStateRecord.from_envelope(envelope)
    expected = format_cdef_id(model.stable_hash())
    if record.subject_cdef_id != expected:
        raise ValueError("stored model state subject does not match the requested model")
    if len(record.storage) != 1 or record.storage[0].kind != "product-dir":
        raise ValueError("managed model state must use one immutable product directory")
    require_product_integrity(store.records, envelope)
    root = store.records.resolve_storage_ref(record.storage[0], record_id=record_id)
    fresh = Repo(store).load_or_build(
        model,
        instance="new",
        cache="none",
        restore_state=False,
    )
    fresh.restore_state_from_dir(str(root))
    return fresh


def write_model_state_output(context, slot: str, model) -> None:
    """Stream a model's state directory to one managed stored-state output."""

    with tempfile.TemporaryDirectory(prefix="dryml-trained-model-") as temp:
        model.save_state_to_dir(temp)
        root = Path(temp)
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            context.write_output(
                slot,
                path.relative_to(root).as_posix(),
                _file_chunks(path),
                representation=MODEL_STATE_REPRESENTATION,
            )


def _snapshot_state_tree(model, source: Path, store) -> LocatedRecordRef:
    subject = format_cdef_id(model.stable_hash())
    for _attempt in range(3):
        before = _tree_manifest(source)
        record = StoredStateRecord(
            subject_cdef_id=subject,
            representation_id=MODEL_STATE_REPRESENTATION["id"],
            storage=(StorageRef.self_product(role="initial-model-state"),),
            state_role="initial-model-state",
            manifest=before.to_json(),
            metadata={"writer": "dryml.models.snapshot_model_state"},
        ).to_envelope()
        attached = attach_record_id(record)
        if store.records.has_record(attached["id"]):
            if before != _tree_manifest(source):
                continue
            require_product_integrity(store.records, store.records.read_record(attached["id"]))
            return LocatedRecordRef(store.catalog_key(), attached["id"])
        with ProductWriteSession(store.records) as session:
            for entry in before.entries:
                src = source / entry.path
                dest = session.staging_dir / entry.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                with src.open("rb") as source_handle, dest.open("xb") as dest_handle:
                    shutil.copyfileobj(source_handle, dest_handle, length=1024 * 1024)
            copied = session.manifest()
            after = _tree_manifest(source)
            if before != copied or before != after:
                continue
            try:
                return session.commit_record(attached).located
            except Exception:
                if store.records.has_record(attached["id"]):
                    require_product_integrity(
                        store.records, store.records.read_record(attached["id"])
                    )
                    return LocatedRecordRef(store.catalog_key(), attached["id"])
                raise
    raise RuntimeError("model state changed while creating an immutable snapshot")


def _has_state_files(root: Path) -> bool:
    return any(
        path.is_file() and path.relative_to(root).as_posix() != "def.pkl"
        for path in root.rglob("*")
    )


def _tree_manifest(root: Path) -> ProductManifest:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        size = 0
        for chunk in _file_chunks(path):
            digest.update(chunk)
            size += len(chunk)
        entries.append(ProductManifestEntry(path.relative_to(root).as_posix(), size, digest.hexdigest()))
    return ProductManifest(tuple(entries))


def _file_chunks(path: Path, size: int = 1024 * 1024):
    with path.open("rb") as handle:
        while chunk := handle.read(size):
            yield chunk


def signature_discovery(obj: Any, **kwargs):
    try:
        from .tf.utils import tf_signature_discovery
        return tf_signature_discovery(obj, **kwargs)
    except (ImportError, ModuleNotFoundError):
        pass

    raise ValueError("Unable to guess a signature based on the object.")


__all__ = [
    "advance_train_state",
    "dataset_is_batched",
    "finite_dataset_len",
    "hydrate_model_state",
    "MODEL_STATE_REPRESENTATION",
    "prepare_training_data",
    "signature_discovery",
    "snapshot_model_state",
    "validate_num_examples",
    "write_model_state_output",
]
