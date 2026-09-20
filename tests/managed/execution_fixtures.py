"""Importable managed-execution fixtures shared by backend integration tests."""

from __future__ import annotations

import functools
from pathlib import Path

from dryml.annotations import Annotation, attach_annotation
from dryml.core import function
from dryml.core.object import Pickleable
from dryml.core.signatures import Mat, Ref
from dryml.core.reference_values import StateRef
from dryml.environments import req as environment_req
from dryml.managed import ManagedConfig, managed_operation
from dryml.worlds import req as world_req


ORDERS = (
    ("operation_afm", ("A", "F", "M")),
    ("operation_amf", ("A", "M", "F")),
    ("operation_fam", ("F", "A", "M")),
    ("operation_fma", ("F", "M", "A")),
    ("operation_maf", ("M", "A", "F")),
    ("operation_mfa", ("M", "F", "A")),
)


def _annotate(target):
    """Attach passive collection and Dispatch declaration evidence."""

    target = attach_annotation(
        target, Annotation("test.integration", "present"),
    )
    return environment_req(source="managed-execution-matrix")(target)


def _matrix_method(written_order):
    """Build one method with its actual written A/F/M order."""

    def operation(self, value: Mat[StateRef], *, managed) -> Mat[StateRef]:
        """Materialize one saved argument and return it through the same boundary."""

        self.calls += 1
        self.value += value.value
        return value

    decorators = {"A": _annotate, "F": function, "M": managed_operation()}
    member = operation
    for decorator in reversed(written_order):
        member = decorators[decorator](member)
    return member


class MatrixArgument(Pickleable):
    """Small stateful materialized argument for all backend matrix placements."""

    def __init__(self, value: int) -> None:
        """Store the scalar observed by a managed matrix method."""

        self.value = value


class ManagedMatrixValue(Pickleable):
    """Importable receiver containing each supported managed decorator order."""

    def __init__(self) -> None:
        """Initialize lifecycle, value, and final-publication observations."""

        self.calls = 0
        self.value = 0
        self.save_calls = 0

    def save_state_to_dir_imp(self, dest_dir, *, codec) -> None:
        """Persist the observable final-save counter with the matrix state."""

        del codec
        self.save_calls += 1
        Path(dest_dir, "matrix-state").write_text(
            f"{self.calls},{self.value},{self.save_calls}", encoding="ascii",
        )

    def restore_state_from_dir_imp(self, src_dir, *, codec) -> None:
        """Restore the matrix observations from one published StateRef."""

        del codec
        values = Path(src_dir, "matrix-state").read_text(
            encoding="ascii",
        ).split(",")
        self.calls, self.value, self.save_calls = map(int, values)

    operation_afm = _matrix_method(("A", "F", "M"))
    operation_amf = _matrix_method(("A", "M", "F"))
    operation_fam = _matrix_method(("F", "A", "M"))
    operation_fma = _matrix_method(("F", "M", "A"))
    operation_maf = _matrix_method(("M", "A", "F"))
    operation_mfa = _matrix_method(("M", "F", "A"))


class ManagedWorldMatrixValue(Pickleable):
    """Managed A/F/M matrix with requirements on descriptor carriers."""

    operation_afm = _matrix_method(("A", "F", "M"))
    operation_amf = _matrix_method(("A", "M", "F"))
    operation_fam = _matrix_method(("F", "A", "M"))
    operation_fma = _matrix_method(("F", "M", "A"))
    operation_maf = _matrix_method(("M", "A", "F"))
    operation_mfa = _matrix_method(("M", "F", "A"))


for _member, _ in ORDERS:
    world_req(
        cpus=1, source="managed-execution-world-matrix",
    )(getattr(ManagedWorldMatrixValue(), _member)._descriptor)
del _member, _


def _recording_wrapper(target):
    """Retain authored wrapper effects around a managed declaration."""

    @functools.wraps(target)
    def wrapped(self, value, *, managed):
        self.events.append("before")
        try:
            return target(self, value, managed=managed) + 10
        finally:
            self.events.append("after")

    return wrapped


class OuterWrappedManagedValue(Pickleable):
    """Fixture for W(M(method)) before any preceding local invocation."""

    def __init__(self) -> None:
        """Initialize wrapper and managed-body observations."""

        self.events = []
        self.value = 0

    @_recording_wrapper
    @managed_operation()
    def advance(self, value: int, *, managed) -> int:
        """Mutate once through the inner managed lifecycle."""

        self.events.append("body")
        self.value += value
        return self.value


class InnerFunctionWrappedManagedValue(Pickleable):
    """Fixture for M(W(F(method))) before any preceding local invocation."""

    def __init__(self) -> None:
        """Initialize wrapper and managed-body observations."""

        self.events = []
        self.value = 0

    @managed_operation()
    @_recording_wrapper
    @function
    def advance(self, value: int, *, managed) -> int:
        """Mutate once within the recognized function handoff."""

        self.events.append("body")
        self.value += value
        return self.value


class DispatchDiscoveryValue(Pickleable):
    """Statically declared managed member invoked by the discovery wrapper."""

    def __init__(self) -> None:
        """Initialize the one managed mutation used by Dispatch placement tests."""

        self.value = 0

    @managed_operation()
    def advance(self, value: int, *, managed) -> int:
        """Mutate once while retaining the declaration-bearing authored method."""

        self.value += value
        return self.value


class ConflictingManagedDiscoveryValue(Pickleable):
    """Managed fixture with incompatible passive environment declarations."""

    def __init__(self) -> None:
        """Initialize the mutation counter used to prove preflight rejection."""

        self.calls = 0

    @managed_operation()
    @environment_req(python=">=4", source="managed-conflict-left")
    @environment_req(python="<3", source="managed-conflict-right")
    def advance(self, *, managed) -> None:
        """Mutate only if Dispatch incorrectly admits the conflicting call."""

        self.calls += 1


class ConflictingManagedWorldDiscoveryValue(Pickleable):
    """Managed fixture with incompatible passive world declarations."""

    def __init__(self) -> None:
        """Initialize the mutation counter used to prove preflight rejection."""

        self.calls = 0

    @world_req(
        cpus={"min": None, "max": 1},
        source="managed-world-conflict-left",
    )
    @world_req(
        cpus={"min": 2, "max": None},
        source="managed-world-conflict-right",
    )
    @managed_operation()
    def advance(self, *, managed) -> None:
        """Mutate only if Dispatch incorrectly admits the conflicting call."""

        self.calls += 1


@environment_req(source="managed-dispatch-discovery")
@function
def declared_managed_call(receiver, value: int) -> int:
    """Carry a passive requirement through Dispatch into a managed member call."""

    return receiver.advance(value)


def nested_managed_call(
        receiver, value: Ref[StateRef], nested_config,
) -> Ref[StateRef]:
    """Use a StateRef at the nested Mat boundary and preserve it for the caller."""

    receiver.operation_afm(
        value, managed=nested_config["config"][0],
    )
    return value


def nested_managed_advance(receiver, value: int, nested_config) -> int:
    """Call a primitive managed member with config held in nested data."""

    return receiver.advance(value, managed=nested_config["config"][0])


def nested_unrelated_function(value: int) -> int:
    """Provide a separate function boundary for normalization-counter coverage."""

    return function(lambda item: item + 1)(value)


def matrix_config(repo, control_store) -> ManagedConfig:
    """Construct explicit selected state and control authority for a matrix call."""

    return ManagedConfig(state_repo=repo, control_store=control_store)
