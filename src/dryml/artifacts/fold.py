"""Deferred managed streaming folds over referenced Datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
import sys
from typing import Any, Generic, TypeVar

import numpy as np

from dryml.core import AutoRef, ConcreteDefinition, Definition, ObjectRef, Ref, StateRef
from dryml.core.tensor_spec import Dynamic, SpecTree, TensorSpec, as_tensor_spec, is_spec_tree, iter_specs, spec_tree_is_batched
from dryml.data import Dataset
from dryml.managed import ManagedContext, managed_operation
from dryml.methods import Accumulator, ImplementationSelectionError, Method
from dryml.methods.signature import satisfies, spec_node

from .value import Value, _VALUE_FORMAT, _VALUE_VERSION


ResultT = TypeVar("ResultT")


class Fold(Value[ResultT], Generic[ResultT]):
    """A deferred, non-resumable managed fold over one referenced Dataset.

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

    Construction and persistence retain only the declared graph; they do not load
    the source, select or invoke Methods, or allocate carry. Every compute attempt
    owns one source iterator and fresh carry. A failure before complete terminal
    validation preserves the prior Value payload. After payload installation, the
    managed lifecycle may still fail publication; in that case readiness remains
    true but the operation honestly raises instead of reporting completion.
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

    @managed_operation(resumable=False)
    def compute(self, *, managed: ManagedContext) -> None:
        """Run one fresh traversal and install its complete terminal result.

        Args:
            managed: Active managed context selecting the Repo used to materialize
                the retained source and publish the Fold state.

        Raises:
            ValueError: If source spec/batch semantics are unavailable, the source
                is empty, or an observation has zero size.
            TypeError: If the retained source does not materialize as a Dataset.
            ImplementationSelectionError: If a declared Method contract cannot be
                selected or its input/output validation fails.
            Exception: Propagates source, Method, result-validation, interruption,
                and managed-publication failures without exposing a partial result.

        Side Effects:
            Materializes the source under the selected Repo, owns and closes one
            source iterator, invokes declared Methods, and atomically installs one
            validated terminal payload before managed final publication.
        """

        source = self._load_source(managed)
        source_spec = _source_spec(source)
        initial_selection = (
            None if _source_requires_refinement(source_spec)
            else self._select_initializer(source_spec)
        )
        iterator = iter(source)
        try:
            first = _first_observation(iterator)
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
            for observation in iterator:
                _validate_nonempty_observation(observation)
                next_observation_spec = _refine_observation_spec(source_spec, observation)
                _require_spec_satisfaction(
                    observed_spec, next_observation_spec, "source observation",
                )
                if _observation_devices(observation) != observation_devices:
                    raise ValueError("Fold source observation device changed during traversal.")
                carry = transition(observation, carry)

            result = carry if finalizer is None else finalizer(carry)
            self._install_value_payload({
                "format": _VALUE_FORMAT,
                "version": _VALUE_VERSION,
                "present": True,
                "result": _normalize_result(result),
            })
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

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
