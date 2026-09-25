"""Explicit observation/carry Method contracts for streaming reductions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Generic, TypeVar

from dryml.core.backend import Backend
from dryml.core.tensor_spec import BatchMode, SpecTree

from .errors import ImplementationSelectionError
from .implementation import MethodImplementation
from .method import Method
from .signature import runtime_facts


ObservationT = TypeVar("ObservationT")
StateT = TypeVar("StateT")


class Accumulator(Method, Generic[ObservationT, StateT]):
    """Abstract logical Method that advances an invocation-owned carry state.

    Implementations receive an observation as their first argument and an
    independent incoming state as their second argument, returning the next
    state. Trait-backed alternatives may satisfy the abstract logical call, but
    subclasses must always provide pure two-input output-spec inference. An
    Accumulator stores no mutable carry on the Method object.
    """

    @abstractmethod
    def __call__(self, observation: ObservationT, state: StateT) -> StateT:
        """Return the next invocation-owned state for one observation.

        Args:
            observation: The current source observation or batch.
            state: The incoming state for this independent invocation.

        Returns:
            A next state compatible with the configured carry contract.

        Raises:
            Implementations may raise backend or reduction-specific errors.
        """

    @abstractmethod
    def infer_output_spec(
        self,
        observation_spec: SpecTree,
        state_spec: SpecTree,
    ) -> SpecTree:
        """Infer the next-state specification without executing a transition.

        Args:
            observation_spec: Normalized specification for the first observation.
            state_spec: Normalized specification for the incoming carry.

        Returns:
            A normalized next-state specification.

        Raises:
            Implementations may raise ``TypeError`` or ``ValueError`` for an
            unsupported observation/carry shape or structure.
        """

    def _runtime_selection_facts(self, args, kwargs):
        """Derive direct-call traits solely from the first observation argument."""

        return runtime_facts(args[:1], {})


class AccumulatorGroup(Accumulator[ObservationT, StateT]):
    """Advance matching accumulator branches over one shared observation.

    Args:
        accumulators: A non-empty list, tuple, or string-keyed mapping of
            Accumulator leaves. The collection defines the required carry and
            result structure.

    Selected group calls select each child once for the provided observation and
    carry specs, then retain the resulting carriers. The group itself owns no
    carry values, so using one child declaration in multiple branches never
    aliases execution state.

    Raises:
        TypeError: If the collection is not a supported structure, has a
            non-string mapping key, or contains a non-Accumulator leaf.
        ValueError: If the collection contains no accumulator leaves.
    """

    def __init__(self, accumulators: object) -> None:
        """Validate and retain the declared accumulator collection."""

        self.accumulators = self._validate_collection(accumulators)

    def __call__(self, observation: ObservationT, state: StateT) -> StateT:
        """Advance every declared branch using its corresponding state entry.

        Args:
            observation: Observation delivered identically to every child.
            state: Carry structure matching ``accumulators``.

        Returns:
            Next-state values in the declared list, tuple, or named mapping
            structure.

        Raises:
            TypeError: If ``state`` does not match the declared structure.

        Side Effects:
            Direct calls use each child's ordinary Method call behavior. Selected
            calls retain child carriers and avoid repeat selection.
        """

        return self._map_state(state, lambda accumulator, carry: accumulator(observation, carry))

    def infer_output_spec(
        self,
        observation_spec: SpecTree,
        state_spec: SpecTree,
    ) -> SpecTree:
        """Infer each child next-state spec while preserving declared structure.

        Args:
            observation_spec: Shared normalized observation specification.
            state_spec: Carry specification matching ``accumulators``.

        Returns:
            A next-state specification with the same declared collection shape.

        Raises:
            TypeError: If the carry specification structure does not match the
            declared accumulator collection.

        Side Effects:
            None. This method does not select or invoke children.
        """

        return self._map_state(
            state_spec,
            lambda accumulator, carry_spec: accumulator.infer_output_spec(observation_spec, carry_spec),
        )

    def find_implementation(
        self,
        input_spec: SpecTree | None = None,
        *additional_input_specs: SpecTree,
        backend: Backend | str | None = None,
        batch_mode: BatchMode | str | None = None,
        output_spec: SpecTree | None = None,
    ) -> MethodImplementation:
        """Select the group and its children once for observation/carry contracts.

        Args:
            input_spec: Required normalized observation specification.
            *additional_input_specs: Exactly one required normalized carry spec.
            backend: Optional observation-driven backend constraint.
            batch_mode: Optional observation-driven batch constraint.
            output_spec: Optional next-state constraint matching the group shape.

        Returns:
            A selected group implementation retaining one selected carrier per
            child branch.

        Raises:
            ImplementationSelectionError: If required specs are absent, their
            structures conflict, or the group/child selection is invalid.

        Side Effects:
            Inspects and binds child implementations but does not invoke them or
            change their eager/learning/cached preparation state.
        """

        if input_spec is None or len(additional_input_specs) != 1:
            raise ImplementationSelectionError("conflict")
        state_spec = additional_input_specs[0]
        implementation = super().find_implementation(
            input_spec,
            state_spec,
            backend=backend,
            batch_mode=batch_mode,
            output_spec=output_spec,
        )
        try:
            selected_children = (
                self._map_state(
                    state_spec,
                    lambda accumulator, carry_spec: accumulator.find_implementation(
                        input_spec,
                        carry_spec,
                        backend=backend,
                        batch_mode=batch_mode,
                    ),
                )
                if output_spec is None
                else self._map_state_with_output(
                    self.accumulators,
                    state_spec,
                    output_spec,
                    lambda accumulator, carry_spec, next_spec: accumulator.find_implementation(
                        input_spec,
                        carry_spec,
                        backend=backend,
                        batch_mode=batch_mode,
                        output_spec=next_spec,
                    ),
                )
            )
        except TypeError as error:
            raise ImplementationSelectionError("conflict") from error

        def invoke_group(observation: ObservationT, state: StateT) -> StateT:
            return self._map_state_with(
                selected_children,
                state,
                lambda selected, carry: selected(observation, carry),
            )

        from dataclasses import replace

        return replace(implementation, _invoker=invoke_group)

    @staticmethod
    def _validate_collection(accumulators: object) -> object:
        """Copy supported collection structure while validating every accumulator leaf."""

        def validate(value: object) -> object:
            if isinstance(value, Mapping):
                if not value:
                    raise ValueError("AccumulatorGroup requires at least one Accumulator.")
                if not all(isinstance(key, str) for key in value):
                    raise TypeError("AccumulatorGroup mapping keys must be strings.")
                return {key: validate(child) for key, child in value.items()}
            if isinstance(value, list):
                if not value:
                    raise ValueError("AccumulatorGroup requires at least one Accumulator.")
                return [validate(child) for child in value]
            if isinstance(value, tuple):
                if not value:
                    raise ValueError("AccumulatorGroup requires at least one Accumulator.")
                return tuple(validate(child) for child in value)
            if not isinstance(value, Accumulator):
                raise TypeError(
                    f"AccumulatorGroup expects Accumulator leaves, got {type(value).__name__}."
                )
            return value

        if not isinstance(accumulators, (Mapping, list, tuple)):
            raise TypeError("AccumulatorGroup requires a list, tuple, or string-keyed mapping.")
        return validate(accumulators)

    def _map_state(self, state: object, fn):
        """Apply ``fn`` to declared leaves and one matching state structure."""

        return self._map_state_with(self.accumulators, state, fn)

    @staticmethod
    def _map_state_with(accumulators: object, state: object, fn):
        """Recursively preserve collection structure while pairing state entries."""

        if isinstance(accumulators, Mapping):
            if not isinstance(state, Mapping) or tuple(accumulators) != tuple(state):
                raise TypeError("AccumulatorGroup state mapping does not match its accumulator names.")
            return {
                key: AccumulatorGroup._map_state_with(accumulator, state[key], fn)
                for key, accumulator in accumulators.items()
            }
        if isinstance(accumulators, tuple):
            if not isinstance(state, tuple) or len(accumulators) != len(state):
                raise TypeError("AccumulatorGroup tuple state does not match its accumulators.")
            return tuple(
                AccumulatorGroup._map_state_with(accumulator, carry, fn)
                for accumulator, carry in zip(accumulators, state)
            )
        if isinstance(accumulators, list):
            if not isinstance(state, list) or len(accumulators) != len(state):
                raise TypeError("AccumulatorGroup list state does not match its accumulators.")
            return [
                AccumulatorGroup._map_state_with(accumulator, carry, fn)
                for accumulator, carry in zip(accumulators, state)
            ]
        return fn(accumulators, state)

    @staticmethod
    def _map_state_with_output(accumulators: object, state: object, output: object, fn):
        """Pair declared leaves with matching carry and next-state structures."""

        if isinstance(accumulators, Mapping):
            if (
                not isinstance(state, Mapping)
                or not isinstance(output, Mapping)
                or tuple(accumulators) != tuple(state)
                or tuple(accumulators) != tuple(output)
            ):
                raise TypeError("AccumulatorGroup output mapping does not match its accumulator names.")
            return {
                key: AccumulatorGroup._map_state_with_output(
                    accumulator, state[key], output[key], fn,
                )
                for key, accumulator in accumulators.items()
            }
        if isinstance(accumulators, tuple):
            if (
                not isinstance(state, tuple)
                or not isinstance(output, tuple)
                or len(accumulators) != len(state)
                or len(accumulators) != len(output)
            ):
                raise TypeError("AccumulatorGroup tuple output does not match its accumulators.")
            return tuple(
                AccumulatorGroup._map_state_with_output(accumulator, carry, next_state, fn)
                for accumulator, carry, next_state in zip(accumulators, state, output)
            )
        if isinstance(accumulators, list):
            if (
                not isinstance(state, list)
                or not isinstance(output, list)
                or len(accumulators) != len(state)
                or len(accumulators) != len(output)
            ):
                raise TypeError("AccumulatorGroup list output does not match its accumulators.")
            return [
                AccumulatorGroup._map_state_with_output(accumulator, carry, next_state, fn)
                for accumulator, carry, next_state in zip(accumulators, state, output)
            ]
        return fn(accumulators, state, output)


__all__ = ["Accumulator", "AccumulatorGroup"]
