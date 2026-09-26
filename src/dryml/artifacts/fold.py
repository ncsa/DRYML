"""Deferred managed streaming folds over referenced Datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
import os
import sys
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from dryml.core import AutoRef, ConcreteDefinition, Definition, ObjectRef, Ref, StateRef
from dryml.core.dtype import normalize_dtype
from dryml.core.tensor_spec import Dynamic, SpecTree, TensorSpec, as_tensor_spec, is_spec_tree, iter_specs, spec_tree_is_batched
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.data import Dataset
from dryml.managed import ManagedContext, managed_operation
from dryml.methods import Accumulator, ImplementationSelectionError, Method
from dryml.methods.signature import satisfies, spec_node

from .value import Value, _VALUE_FORMAT, _VALUE_VERSION


ResultT = TypeVar("ResultT")
_PROGRESS_FILENAME = "fold-progress.pkl"
_PROGRESS_FORMAT = "dryml.artifacts.fold-progress"
_PROGRESS_VERSION = 1
_PROGRESS_KEYS = frozenset((
    "state", "count", "carry", "source_binding", "source_spec",
    "observed_spec", "carry_spec", "observation_devices", "program",
    "backend_devices", "operation_id", "attempt_id",
))


class Fold(Value[ResultT], Generic[ResultT]):
    """A deferred, checkpoint-resumable managed fold over one referenced Dataset.

    Args:
        src: Non-materializing source reference selected through ``Ref[AutoRef]``.
            The source is loaded only during :meth:`compute` using that invocation's
            selected state Repo.
        initial_state: Declared unary Method that creates fresh invocation-local
            carry from the first actual observation or batch.
        accumulator: Declared two-input Accumulator that advances the carry once
            for every source observation or batch.
        finalize: Optional declared unary Method converting final carry to a result.
            When omitted, the final carry is the result.

    Construction retains only the declared graph and zero progress; it does not
    load the source, select or invoke Methods, or allocate carry. Checkpoints store
    bounded host carry plus yielded position and compatibility evidence, never an
    iterator or selected callable. A failure before complete terminal validation
    preserves the prior Value payload. After payload installation, the managed
    lifecycle may still fail publication; readiness then remains true while the
    operation honestly raises instead of reporting completion.
    """

    def __init__(
        self,
        src: Ref[AutoRef],
        *,
        initial_state: Method,
        accumulator: Accumulator,
        finalize: Method | None = None,
    ) -> None:
        """Retain an inert source reference and all declared Method dependencies.

        Raises:
            TypeError: If declared roles are not Method/Accumulator instances.

        Side Effects:
            None. This constructor intentionally does not materialize ``src`` or
            select, invoke, or prepare any declared Method.
        """

        if not isinstance(initial_state, Method):
            raise TypeError("Fold initial_state must be a Method.")
        if not isinstance(accumulator, Accumulator):
            raise TypeError("Fold accumulator must be an Accumulator.")
        if finalize is not None and not isinstance(finalize, Method):
            raise TypeError("Fold finalize must be a Method or None.")
        self.src = src
        self.initial_state = initial_state
        self.accumulator = accumulator
        self.finalize = finalize
        self._processed_count = 0
        self._fold_progress = None

    @property
    def processed_count(self) -> int:
        """Return the number of source yields in terminal or checkpointed state."""

        return int(getattr(self, "_processed_count", 0))

    @property
    def ready(self) -> bool:
        """Return whether a complete validated terminal result is installed.

        Returns:
            ``True`` only after :meth:`compute` has fully normalized and installed
            its terminal Value payload, or that payload was restored.

        Side Effects:
            Does not compute or materialize the source reference.
        """

        return self._value_is_present()

    @managed_operation(resumable=True, store_parameter="store")
    def compute(
            self, *, checkpoint_every: int = 1000, store=None,
            managed: ManagedContext) -> None:
        """Run or resume one traversal and install its complete terminal result.

        Args:
            checkpoint_every: Positive exact number of successful source-yield
                transitions between managed progress checkpoints.
            store: Optional managed publication Store override applying to both
                progress checkpoints and final publication.
            managed: Active managed context selecting the Repo used to materialize
                the retained source and publish the Fold state.

        Raises:
            ValueError: If the checkpoint cadence, progress evidence, source
                spec/batch semantics, or observations violate the Fold contract.
            TypeError: If the retained source does not materialize as a Dataset.
            ImplementationSelectionError: If a declared Method contract cannot be
                selected or its input/output validation fails.
            Exception: Propagates source, Method, result-validation, interruption,
                and managed-publication failures without exposing a partial result.

        Side Effects:
            Materializes a fresh source cursor unless resuming exhausted progress,
            checkpoints host carry and position at safe points, and atomically
            installs one validated terminal payload before final publication.
        """

        checkpoint_every = _validate_checkpoint_every(checkpoint_every)
        if managed.is_resuming:
            progress = self._resume_progress(managed)
            source_spec = _decode_spec_tree(progress["source_spec"])
            observed_spec = _decode_spec_tree(progress["observed_spec"])
            carry_spec = _decode_spec_tree(progress["carry_spec"])
            carry = _restore_carry(progress["carry"], carry_spec)
            if _encode_spec_tree(_runtime_spec(carry)) != progress["carry_spec"]:
                raise ValueError("Fold checkpoint carry does not match its saved specification.")
            initial_spec, _ = self._select_initializer(observed_spec)
            _require_spec_satisfaction(initial_spec, carry_spec, "initializer carry")
            transition, finalizer = self._select_followups(observed_spec, carry_spec)
            count = progress["count"]
            if progress["state"] == "exhausted":
                self._install_terminal(carry, finalizer, count)
                return

            source = self._load_source(managed)
            if _encode_spec_tree(_source_spec(source)) != progress["source_spec"]:
                raise ValueError("Fold checkpoint source specification is incompatible.")
            cursor = source.iterator()
            try:
                cursor.skip(count)
                self._continue(
                    cursor, source_spec, observed_spec, progress["observation_devices"],
                    carry_spec, carry, transition, finalizer, count,
                    checkpoint_every, managed,
                )
            finally:
                cursor.close()
            return

        self._processed_count = 0
        self._fold_progress = None
        source = self._load_source(managed)
        source_spec = _source_spec(source)
        initial_selection = (
            None if _source_requires_refinement(source_spec)
            else self._select_initializer(source_spec)
        )
        cursor = source.iterator()
        try:
            first = _first_observation(cursor)
            _validate_nonempty_observation(first)
            observed_spec = _refine_observation_spec(source_spec, first)
            observation_devices = _observation_devices(first)
            initial_spec, initializer = (
                self._select_initializer(observed_spec)
                if initial_selection is None else initial_selection
            )

            # The initializer sees the actual allocation prototype; the transition
            # then consumes this same value without a second traversal or peek.
            carry = initializer(first)
            carry_spec = _runtime_spec(carry)
            _require_spec_satisfaction(initial_spec, carry_spec, "initializer carry")
            transition, finalizer = self._select_followups(
                observed_spec, carry_spec,
            )
            carry = transition(first, carry)
            count = 1
            self._processed_count = count
            if count % checkpoint_every == 0:
                self._checkpoint_progress(
                    "active", count, carry, source_spec, observed_spec, carry_spec,
                    observation_devices, managed,
                )
            self._continue(
                cursor, source_spec, observed_spec, observation_devices,
                carry_spec, carry, transition, finalizer, count,
                checkpoint_every, managed,
            )
        finally:
            cursor.close()

    def _continue(
            self, cursor, source_spec, observed_spec, observation_devices,
            carry_spec, carry, transition, finalizer, count, checkpoint_every,
            managed) -> None:
        """Consume a positioned cursor through exhaustion and finalize once."""

        exhaustion_close_error = None
        while True:
            try:
                observation = next(cursor)
            except StopIteration:
                break
            except BaseException as error:
                # DatasetCursor marks itself closed only after its source reports
                # EOF. Preserve Fold's established finalize-then-report ordering
                # when releasing that exhausted source is what failed.
                if not getattr(cursor, "_closed", False):
                    raise
                exhaustion_close_error = error
                break
            _validate_nonempty_observation(observation)
            next_observation_spec = _refine_observation_spec(source_spec, observation)
            _require_spec_satisfaction(
                observed_spec, next_observation_spec, "source observation",
            )
            if _observation_devices(observation) != observation_devices:
                raise ValueError("Fold source observation device changed during traversal.")
            carry = transition(observation, carry)
            count += 1
            self._processed_count = count
            if count % checkpoint_every == 0:
                self._checkpoint_progress(
                    "active", count, carry, source_spec, observed_spec, carry_spec,
                    observation_devices, managed,
                )

        self._checkpoint_progress(
            "exhausted", count, carry, source_spec, observed_spec, carry_spec,
            observation_devices, managed,
        )
        self._install_terminal(carry, finalizer, count)
        if exhaustion_close_error is not None:
            raise exhaustion_close_error

    def _checkpoint_progress(
            self, state, count, carry, source_spec, observed_spec, carry_spec,
            observation_devices, managed) -> None:
        """Install one self-consistent host progress snapshot and publish it."""

        carry_record = _capture_carry(carry, carry_spec)
        progress = {
            "state": state,
            "count": count,
            "carry": carry_record,
            "source_binding": _source_binding_evidence(self.src),
            "source_spec": _encode_spec_tree(source_spec),
            "observed_spec": _encode_spec_tree(observed_spec),
            "carry_spec": _encode_spec_tree(carry_spec),
            "observation_devices": tuple(observation_devices),
            "program": self._program_evidence(),
            "backend_devices": _carry_backend_devices(carry_record),
            "operation_id": managed.operation_id,
            "attempt_id": managed.attempt_id,
        }
        self._fold_progress = progress
        self._processed_count = count
        managed.checkpoint()

    def _resume_progress(self, managed: ManagedContext) -> dict[str, Any]:
        """Validate restored progress against this invocation before carry use."""

        progress = getattr(self, "_fold_progress", None)
        if progress is None:
            raise ValueError("Fold resume has no saved progress.")
        if (
                progress["operation_id"] != managed.operation_id
                or progress["attempt_id"] != managed.attempt_id
                or progress["source_binding"] != _source_binding_evidence(self.src)
                or progress["program"] != self._program_evidence()
        ):
            raise ValueError("Fold checkpoint belongs to an incompatible source, program, or attempt.")
        return progress

    def _program_evidence(self) -> dict[str, str | None]:
        """Return stable definition identities for all declared Fold roles."""

        return {
            "initializer": self.initial_state.definition.stable_hash(),
            "accumulator": self.accumulator.definition.stable_hash(),
            "finalizer": None if self.finalize is None else self.finalize.definition.stable_hash(),
        }

    def _install_terminal(self, carry, finalizer, count: int) -> None:
        """Install a normalized completed Value, then clear active progress."""

        result = carry if finalizer is None else finalizer(carry)
        self._install_value_payload({
            "format": _VALUE_FORMAT,
            "version": _VALUE_VERSION,
            "present": True,
            "result": _normalize_result(result),
        })
        self._processed_count = count
        self._fold_progress = None

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Persist terminal count and optional active progress separately from Value."""

        pickle_save({
            "format": _PROGRESS_FORMAT,
            "version": _PROGRESS_VERSION,
            "processed_count": self.processed_count,
            "progress": getattr(self, "_fold_progress", None),
        }, os.path.join(dest_dir, _PROGRESS_FILENAME))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Restore validated Fold progress without materializing source or Methods."""

        payload = pickle_load(Path(src_dir, _PROGRESS_FILENAME))
        count, progress = _validate_progress_payload(payload)
        self._processed_count = count
        self._fold_progress = progress

    def _select_initializer(self, observation_spec: SpecTree):
        """Infer and select the initializer before allocating a carry.

        Args:
            observation_spec: Source element or batch contract, possibly carrying
                dynamic dimensions but with explicit batch semantics.

        Returns:
            The inferred initializer carry spec and selected initializer carrier.

        Raises:
            ImplementationSelectionError: If any declared role has no compatible
                implementation under the independent observation/carry contracts.

        Side Effects:
            Performs pure Method inference and local implementation selection only;
            it does not invoke Methods or change their preparation state.
        """

        initial_spec = self.initial_state.infer_output_spec(observation_spec)
        if not is_spec_tree(initial_spec):
            raise ImplementationSelectionError("conflict")
        initializer = self.initial_state.find_implementation(
            observation_spec, output_spec=initial_spec,
        )
        return initial_spec, initializer

    def _select_followups(self, observation_spec: SpecTree, carry_spec: SpecTree):
        """Select fixed-carry transition and finalization carriers after initialization.

        Args:
            observation_spec: First-observation facts refined from the declared
                source contract while retaining its batch semantics.
            carry_spec: Concrete specification inferred from the initialized
                runtime carry.

        Returns:
            Selected transition and optional selected finalizer.

        Raises:
            ImplementationSelectionError: If the initialized carry cannot remain
                fixed under the declared transition/finalizer contracts.

        Side Effects:
            Performs pure inference and local selection only. It never invokes a
            Method or retains the invocation-owned carry.
        """

        next_spec = self.accumulator.infer_output_spec(observation_spec, carry_spec)
        if not is_spec_tree(next_spec):
            raise ImplementationSelectionError("conflict")
        if not satisfies(spec_node(carry_spec), spec_node(next_spec)):
            raise ImplementationSelectionError("conflict")
        transition = self.accumulator.find_implementation(
            observation_spec, carry_spec, output_spec=carry_spec,
        )
        if self.finalize is None:
            return transition, None
        final_spec = self.finalize.infer_output_spec(carry_spec)
        if not is_spec_tree(final_spec):
            raise ImplementationSelectionError("conflict")
        finalizer = self.finalize.find_implementation(carry_spec, output_spec=final_spec)
        return transition, finalizer

    def _load_source(self, managed: ManagedContext) -> Dataset:
        """Materialize one retained source reference through selected Repo authority.

        Raises:
            TypeError: If reference materialization does not produce a Dataset.

        Side Effects:
            May materialize the referenced source through the invocation's borrowed
            Repo. It never closes that Repo, its Stores, or the source itself.
        """

        source_ref = self.src
        repo = managed.state_repo
        if isinstance(source_ref, StateRef):
            source = repo.load_state_ref(source_ref)
        elif isinstance(source_ref, ObjectRef):
            source = repo.build_object_ref(source_ref)
        elif isinstance(source_ref, ConcreteDefinition):
            source = repo._load_structural(source_ref, require_store=False)
        elif isinstance(source_ref, Definition):
            source = repo._load_structural(source_ref.concretize(repo=repo), require_store=False)
        else:
            raise TypeError("Fold source reference is not a supported DRYML reference.")
        if not isinstance(source, Dataset):
            raise TypeError("Fold source reference must materialize as a Dataset.")
        return source

    def _validate_value_result(self, result: ResultT) -> None:
        """Validate Fold's lightweight terminal result tree before installation.

        Args:
            result: Fully completed host-normalized terminal result.

        Raises:
            ValueError: If the result contains an unsupported value or collection
                structure that cannot be persisted as a lightweight Fold payload.

        Side Effects:
            None. Validation is deliberately side-effect-free for atomic payload
            installation and exact-state restoration.
        """

        _validate_lightweight_result(result)


def _validate_checkpoint_every(value: int) -> int:
    """Return one positive exact checkpoint cadence."""

    if type(value) is not int:
        raise TypeError("checkpoint_every must be a positive exact int.")
    if value <= 0:
        raise ValueError("checkpoint_every must be positive.")
    return value


def _source_binding_evidence(source: object) -> dict[str, str]:
    """Encode one inert source binding identity without retaining its graph."""

    if isinstance(source, StateRef):
        return {"kind": "state-ref", "digest": source.digest()}
    if isinstance(source, ObjectRef):
        return {"kind": "object-ref", "digest": source.digest()}
    if isinstance(source, ConcreteDefinition):
        return {"kind": "concrete-definition", "digest": source.stable_hash()}
    if isinstance(source, Definition):
        return {"kind": "definition", "digest": source.stable_hash()}
    raise TypeError("Fold source reference is not a supported DRYML reference.")


def _encode_tree_key(value: object) -> object:
    """Encode common immutable SpecTree keys without persisting arbitrary objects."""

    if value is None or type(value) in {bool, int, str, bytes}:
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ("float", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_encode_tree_key(item) for item in value))
    raise ValueError("Fold checkpoint SpecTree contains an unsupported mapping key.")


def _decode_tree_key(value: object) -> object:
    """Decode one bounded SpecTree mapping key record."""

    if not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not str:
        raise ValueError("Fold checkpoint SpecTree mapping key is malformed.")
    kind, item = value
    expected = {"NoneType": type(None), "bool": bool, "int": int, "str": str, "bytes": bytes}
    if kind in expected and type(item) is expected[kind]:
        return item
    if kind == "float" and type(item) is float and math.isfinite(item):
        return item
    if kind == "tuple" and isinstance(item, tuple):
        return tuple(_decode_tree_key(child) for child in item)
    raise ValueError("Fold checkpoint SpecTree mapping key is unsupported.")


def _encode_dimension(value: object) -> object:
    """Encode one TensorSpec dimension in a pickle-independent scalar form."""

    if value is Dynamic:
        return "dynamic"
    if type(value) is int and value >= 0:
        return value
    raise ValueError("Fold checkpoint TensorSpec dimension is invalid.")


def _decode_dimension(value: object) -> object:
    """Decode one TensorSpec dimension from progress evidence."""

    if type(value) is str and value == "dynamic":
        return Dynamic
    if type(value) is int and value >= 0:
        return value
    raise ValueError("Fold checkpoint TensorSpec dimension is malformed.")


def _encode_spec_tree(spec: SpecTree) -> dict[str, object]:
    """Encode a complete SpecTree as bounded plain compatibility evidence."""

    if isinstance(spec, TensorSpec):
        return {
            "kind": "tensor",
            "dtype": spec.dtype.name,
            "shape": None if spec.shape is None else tuple(_encode_dimension(item) for item in spec.shape),
            "batch": None if spec.batch is None else _encode_dimension(spec.batch),
            "backend": None if spec.backend is None else spec.backend.value,
            "layout": spec.layout.value,
            "axis_names": spec.axis_names,
            "batch_axis_name": spec.batch_axis_name,
            "ragged_rank": spec.ragged_rank,
            "row_splits_dtype": None if spec.row_splits_dtype is None else spec.row_splits_dtype.name,
            "sparse_format": spec.sparse_format,
        }
    if isinstance(spec, Mapping):
        return {
            "kind": "dict",
            "items": tuple(
                (_encode_tree_key(key), _encode_spec_tree(child))
                for key, child in spec.items()
            ),
        }
    if isinstance(spec, tuple):
        return {"kind": "tuple", "items": tuple(_encode_spec_tree(item) for item in spec)}
    if isinstance(spec, list):
        return {"kind": "list", "items": tuple(_encode_spec_tree(item) for item in spec)}
    raise ValueError("Fold checkpoint contains an invalid SpecTree.")


def _decode_spec_tree(record: object) -> SpecTree:
    """Decode and validate one persisted Fold SpecTree evidence record."""

    if not isinstance(record, dict) or type(record.get("kind")) is not str:
        raise ValueError("Fold checkpoint SpecTree is malformed.")
    kind = record["kind"]
    if kind == "tensor":
        fields = {
            "kind", "dtype", "shape", "batch", "backend", "layout", "axis_names",
            "batch_axis_name", "ragged_rank", "row_splits_dtype", "sparse_format",
        }
        if set(record) != fields:
            raise ValueError("Fold checkpoint TensorSpec is malformed.")
        shape = record["shape"]
        if shape is not None and not isinstance(shape, tuple):
            raise ValueError("Fold checkpoint TensorSpec shape is malformed.")
        try:
            return TensorSpec(
                record["dtype"],
                shape=None if shape is None else tuple(_decode_dimension(item) for item in shape),
                batch=None if record["batch"] is None else _decode_dimension(record["batch"]),
                backend=record["backend"],
                layout=record["layout"],
                axis_names=record["axis_names"],
                batch_axis_name=record["batch_axis_name"],
                ragged_rank=record["ragged_rank"],
                row_splits_dtype=record["row_splits_dtype"],
                sparse_format=record["sparse_format"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Fold checkpoint TensorSpec is invalid.") from error
    if kind not in {"dict", "tuple", "list"} or set(record) != {"kind", "items"}:
        raise ValueError("Fold checkpoint SpecTree container is malformed.")
    items = record["items"]
    if not isinstance(items, tuple):
        raise ValueError("Fold checkpoint SpecTree items are malformed.")
    if kind == "dict":
        result = {}
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Fold checkpoint SpecTree mapping item is malformed.")
            key = _decode_tree_key(item[0])
            if key in result:
                raise ValueError("Fold checkpoint SpecTree repeats a mapping key.")
            result[key] = _decode_spec_tree(item[1])
        return result
    values = tuple(_decode_spec_tree(item) for item in items)
    return values if kind == "tuple" else list(values)


def _capture_carry(value: object, spec: SpecTree) -> dict[str, object]:
    """Copy native numerical carry to a tagged lossless host representation."""

    if isinstance(spec, TensorSpec):
        if spec.backend is None or spec.backend.value not in {"numpy", "torch", "tf"}:
            raise ValueError("Fold checkpoints require NumPy, Torch CPU, or TensorFlow CPU carry.")
        backend = spec.backend.value
        device = "host"
        if backend == "torch":
            import dryml.torch  # noqa: F401
            import torch

            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError("Fold checkpoints require Torch carry on CPU.")
            device = "cpu"
            tensor = value.detach().contiguous()
            if tensor.dtype == torch.bfloat16:
                array = tensor.view(torch.uint16).numpy().view(spec.dtype.np())
            else:
                array = tensor.numpy()
        elif backend == "tf":
            import dryml.tf  # noqa: F401
            import tensorflow as tf

            if not isinstance(value, tf.Tensor) or (value.device and "CPU" not in value.device.upper()):
                raise ValueError("Fold checkpoints require TensorFlow carry on CPU.")
            device = "cpu"
            array = value.numpy()
        else:
            array = np.asarray(value)
        try:
            dtype = normalize_dtype(array.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Fold checkpoint carry dtype is unsupported.") from error
        shape = tuple(int(item) for item in array.shape)
        if dtype != spec.dtype or shape != spec.shape or dtype.kind not in {
                "bool", "int", "uint", "float", "bfloat", "complex",
        }:
            raise ValueError("Fold checkpoint carry does not match its numerical specification.")
        return {
            "kind": "tensor", "backend": backend, "device": device,
            "dtype": dtype.name, "shape": shape,
            "data": np.array(array, copy=True, order="C"),
        }
    if isinstance(spec, Mapping):
        if not isinstance(value, Mapping) or tuple(spec) != tuple(value):
            raise ValueError("Fold checkpoint carry structure does not match its specification.")
        return {
            "kind": "dict",
            "items": tuple(
                (_encode_tree_key(key), _capture_carry(value[key], child))
                for key, child in spec.items()
            ),
        }
    if isinstance(spec, tuple):
        if not isinstance(value, tuple) or len(value) != len(spec):
            raise ValueError("Fold checkpoint tuple carry does not match its specification.")
        return {"kind": "tuple", "items": tuple(_capture_carry(item, child) for item, child in zip(value, spec))}
    if isinstance(spec, list):
        if not isinstance(value, list) or len(value) != len(spec):
            raise ValueError("Fold checkpoint list carry does not match its specification.")
        return {"kind": "list", "items": tuple(_capture_carry(item, child) for item, child in zip(value, spec))}
    raise ValueError("Fold checkpoint carry specification is invalid.")


def _restore_carry(record: object, spec: SpecTree, *, materialize: bool = True) -> object:
    """Restore tagged host carry to its saved native numerical backend."""

    if not isinstance(record, dict) or type(record.get("kind")) is not str:
        raise ValueError("Fold checkpoint carry is malformed.")
    kind = record["kind"]
    if isinstance(spec, TensorSpec):
        fields = {"kind", "backend", "device", "dtype", "shape", "data"}
        if kind != "tensor" or set(record) != fields or not isinstance(record["data"], np.ndarray):
            raise ValueError("Fold checkpoint tensor carry is malformed.")
        backend = None if spec.backend is None else spec.backend.value
        expected_device = "host" if backend == "numpy" else "cpu"
        data = record["data"]
        try:
            dtype = normalize_dtype(data.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Fold checkpoint tensor dtype is unsupported.") from error
        if (
                record["backend"] != backend or record["device"] != expected_device
                or record["dtype"] != spec.dtype.name or dtype != spec.dtype
                or record["shape"] != spec.shape or tuple(data.shape) != spec.shape
        ):
            raise ValueError("Fold checkpoint tensor evidence is incompatible.")
        if not materialize:
            return None
        data = np.array(data, copy=True, order="C")
        if backend == "numpy":
            return data
        if backend == "torch":
            import dryml.torch  # noqa: F401
            import torch

            if data.size == 0:
                return torch.empty(spec.shape, dtype=spec.dtype.torch(), device="cpu")
            raw = bytearray(data.tobytes())
            return torch.frombuffer(raw, dtype=spec.dtype.torch()).clone().reshape(spec.shape)
        if backend == "tf":
            import dryml.tf  # noqa: F401
            import tensorflow as tf

            with tf.device("/CPU:0"):
                return tf.convert_to_tensor(data, dtype=spec.dtype.tf())
        raise ValueError("Fold checkpoint carry backend is unsupported.")
    if kind not in {"dict", "tuple", "list"} or set(record) != {"kind", "items"}:
        raise ValueError("Fold checkpoint carry container is malformed.")
    items = record["items"]
    if not isinstance(items, tuple):
        raise ValueError("Fold checkpoint carry items are malformed.")
    if isinstance(spec, Mapping):
        if kind != "dict" or len(items) != len(spec):
            raise ValueError("Fold checkpoint mapping carry is incompatible.")
        result = {}
        for (saved_key, child), (expected_key, child_spec) in zip(items, spec.items()):
            key = _decode_tree_key(saved_key)
            if key != expected_key or key in result:
                raise ValueError("Fold checkpoint mapping carry keys are incompatible.")
            result[key] = _restore_carry(child, child_spec, materialize=materialize)
        return result if materialize else None
    if isinstance(spec, tuple):
        if kind != "tuple" or len(items) != len(spec):
            raise ValueError("Fold checkpoint tuple carry is incompatible.")
        values = tuple(
            _restore_carry(item, child, materialize=materialize)
            for item, child in zip(items, spec)
        )
        return values if materialize else None
    if isinstance(spec, list):
        if kind != "list" or len(items) != len(spec):
            raise ValueError("Fold checkpoint list carry is incompatible.")
        values = [
            _restore_carry(item, child, materialize=materialize)
            for item, child in zip(items, spec)
        ]
        return values if materialize else None
    raise ValueError("Fold checkpoint carry specification is invalid.")


def _carry_backend_devices(record: object) -> tuple[tuple[str, str, str, tuple[int, ...]], ...]:
    """Return ordered per-leaf backend/device/dtype/shape checkpoint evidence."""

    if not isinstance(record, dict):
        raise ValueError("Fold checkpoint carry is malformed.")
    if record.get("kind") == "tensor":
        return ((record["backend"], record["device"], record["dtype"], record["shape"]),)
    items = record.get("items")
    if not isinstance(items, tuple):
        raise ValueError("Fold checkpoint carry items are malformed.")
    children = (item[1] for item in items) if record.get("kind") == "dict" else items
    return tuple(evidence for child in children for evidence in _carry_backend_devices(child))


def _validate_progress_payload(payload: object) -> tuple[int, dict[str, Any] | None]:
    """Validate one complete v1 Fold progress envelope before assignment."""

    fields = {"format", "version", "processed_count", "progress"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Fold progress payload must be a complete v1 envelope.")
    if (
            payload["format"] != _PROGRESS_FORMAT
            or type(payload["version"]) is not int
            or payload["version"] != _PROGRESS_VERSION
    ):
        raise ValueError("Fold progress payload format or version is unsupported.")
    count = payload["processed_count"]
    if type(count) is not int or count < 0:
        raise ValueError("Fold processed count is invalid.")
    progress = payload["progress"]
    if progress is None:
        return count, None
    if not isinstance(progress, dict) or set(progress) != _PROGRESS_KEYS:
        raise ValueError("Fold active progress is malformed.")
    if progress["state"] not in {"active", "exhausted"}:
        raise ValueError("Fold active progress state is unsupported.")
    if type(progress["count"]) is not int or progress["count"] <= 0 or progress["count"] != count:
        raise ValueError("Fold active progress count is inconsistent.")
    if not isinstance(progress["source_binding"], dict) or set(progress["source_binding"]) != {"kind", "digest"}:
        raise ValueError("Fold checkpoint source binding is malformed.")
    if not all(type(progress["source_binding"][name]) is str for name in ("kind", "digest")):
        raise ValueError("Fold checkpoint source binding is invalid.")
    for name in ("source_spec", "observed_spec", "carry_spec"):
        _decode_spec_tree(progress[name])
    if not isinstance(progress["observation_devices"], tuple) or not all(
            item is None or type(item) is str for item in progress["observation_devices"]):
        raise ValueError("Fold checkpoint observation device evidence is malformed.")
    program = progress["program"]
    if not isinstance(program, dict) or set(program) != {"initializer", "accumulator", "finalizer"}:
        raise ValueError("Fold checkpoint program evidence is malformed.")
    if not all(type(program[name]) is str for name in ("initializer", "accumulator")):
        raise ValueError("Fold checkpoint program evidence is invalid.")
    if program["finalizer"] is not None and type(program["finalizer"]) is not str:
        raise ValueError("Fold checkpoint finalizer evidence is invalid.")
    if not all(type(progress[name]) is str and progress[name] for name in ("operation_id", "attempt_id")):
        raise ValueError("Fold checkpoint attempt evidence is invalid.")
    carry_spec = _decode_spec_tree(progress["carry_spec"])
    _restore_carry(progress["carry"], carry_spec, materialize=False)
    if progress["backend_devices"] != _carry_backend_devices(progress["carry"]):
        raise ValueError("Fold checkpoint backend/device evidence is inconsistent.")
    return count, progress


def _source_spec(source: Dataset) -> SpecTree:
    """Return a complete source spec with explicit, uniform batch semantics."""

    try:
        spec = source.spec
    except (AttributeError, ValueError) as error:
        raise ValueError("Fold source has no usable element spec.") from error
    if not is_spec_tree(spec) or not tuple(iter_specs(spec)):
        raise ValueError("Fold source spec must contain at least one TensorSpec.")
    batches = {item.batched for item in iter_specs(spec)}
    if len(batches) != 1:
        raise ValueError("Fold source spec must use one explicit batch meaning.")
    return spec


def _first_observation(iterator):
    """Read one required source observation without opening another iterator."""

    try:
        return next(iterator)
    except StopIteration as error:
        raise ValueError("Fold cannot compute an empty source.") from error


def _validate_nonempty_observation(observation: object) -> None:
    """Reject zero-sized tensor leaves before initializer or transition execution."""

    if isinstance(observation, Mapping):
        values = observation.values()
    elif isinstance(observation, (tuple, list)):
        values = observation
    else:
        values = (observation,)
    for value in values:
        if isinstance(value, (Mapping, tuple, list)):
            _validate_nonempty_observation(value)
        elif _shape_has_zero_dimension(value):
            raise ValueError("Fold source observation must not have zero size.")


def _normalize_result(value: Any) -> Any:
    """Convert a completed native result to one lightweight persistence tree."""

    value = _terminal_host_value(value)
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Fold result contains a non-finite numeric value.")
        return value
    if isinstance(value, np.generic):
        return _normalize_result(value.item())
    if isinstance(value, np.ndarray):
        _validate_numeric_array(value)
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_normalize_result(item) for item in value)
    if isinstance(value, list):
        return [_normalize_result(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _normalize_result(item) for key, item in value.items()}
    raise ValueError(f"Fold result contains unsupported value type {type(value).__name__}.")


def _validate_lightweight_result(value: Any) -> None:
    """Reject restored payloads that are not already lightweight numeric results."""

    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Fold result contains a non-finite numeric value.")
        return
    if isinstance(value, np.generic):
        _validate_lightweight_result(value.item())
        return
    if isinstance(value, np.ndarray):
        _validate_numeric_array(value)
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_lightweight_result(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_lightweight_result(item)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _validate_lightweight_result(item)
        return
    raise ValueError(f"Fold result contains unsupported value type {type(value).__name__}.")


def _shape_has_zero_dimension(value: object) -> bool:
    """Report an empty tensor from native shape metadata without host conversion."""

    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    try:
        return any(dimension == 0 for dimension in shape)
    except TypeError:
        return False


def _refine_observation_spec(source_spec: SpecTree, observation: object) -> SpecTree:
    """Concretize first-item facts while retaining the declared batch contract."""

    actual = as_tensor_spec(observation, batched=spec_tree_is_batched(source_spec))

    def refine(declared: SpecTree, observed: SpecTree) -> SpecTree:
        if isinstance(declared, TensorSpec):
            if not isinstance(observed, TensorSpec):
                raise ValueError("Fold source observation structure does not match its spec.")
            _require_spec_satisfaction(declared, observed, "source observation")
            return replace(
                observed,
                batch=declared.batch,
                batch_axis_name=declared.batch_axis_name if declared.batch is not None else None,
            )
        if isinstance(declared, Mapping):
            if not isinstance(observed, Mapping) or tuple(declared) != tuple(observed):
                raise ValueError("Fold source observation structure does not match its spec.")
            return {key: refine(declared[key], observed[key]) for key in declared}
        if isinstance(declared, tuple):
            if not isinstance(observed, tuple) or len(declared) != len(observed):
                raise ValueError("Fold source observation structure does not match its spec.")
            return tuple(refine(expected, actual) for expected, actual in zip(declared, observed))
        if isinstance(declared, list):
            if not isinstance(observed, list) or len(declared) != len(observed):
                raise ValueError("Fold source observation structure does not match its spec.")
            return [refine(expected, actual) for expected, actual in zip(declared, observed)]
        raise ValueError("Fold source spec is invalid.")

    return refine(source_spec, actual)


def _source_requires_refinement(source_spec: SpecTree) -> bool:
    """Report whether source facts need a real prototype before Method selection."""

    return any(
        spec.backend is None
        or spec.shape is None
        or any(dimension is Dynamic for dimension in spec.shape)
        for spec in iter_specs(source_spec)
    )


def _runtime_spec(value: object) -> SpecTree:
    """Infer the unbatched concrete runtime carry specification after initialization."""

    try:
        return as_tensor_spec(value, batched=False)
    except (TypeError, ValueError) as error:
        raise ImplementationSelectionError("conflict") from error


def _require_spec_satisfaction(expected: SpecTree, actual: SpecTree, role: str) -> None:
    """Require one runtime spec tree to satisfy a declared Fold role contract."""

    try:
        valid = satisfies(spec_node(expected), spec_node(actual))
    except TypeError as error:
        raise ValueError(f"Fold {role} does not match its declared spec.") from error
    if not valid:
        raise ValueError(f"Fold {role} does not match its declared spec.")


def _observation_devices(observation: object) -> tuple[str | None, ...]:
    """Return native device facts for stability checks without moving tensor data."""

    devices: list[str | None] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, "shape"):
            device = getattr(value, "device", None)
            devices.append(None if device is None else str(device))

    visit(observation)
    return tuple(devices)


def _terminal_host_value(value: Any) -> Any:
    """Convert only known loaded CPU tensor types at Fold's terminal boundary."""

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None and isinstance(value, tensorflow.Tensor):
        return value.numpy()
    return value


def _validate_numeric_array(value: np.ndarray) -> None:
    """Validate one already-host array without allocating another array copy."""

    if value.dtype == object or not (
            np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError("Fold result array must have a numeric non-object dtype.")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise ValueError("Fold result array must not use a complex dtype.")
    if np.issubdtype(value.dtype, np.inexact) and not bool(np.all(np.isfinite(value))):
        raise ValueError("Fold result array contains non-finite numeric values.")


Fold.__module__ = "dryml.artifacts"

__all__ = ["Fold"]
