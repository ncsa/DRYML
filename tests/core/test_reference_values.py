from uuid import UUID
import subprocess
import sys

import pytest

from dryml.core import Definition, ObjectId, ObjectRef, StateRef
from dryml.core.object import Object, Serializable
from dryml.core.utils.graph.path import GraphPath


class ReferenceEphemeral(Object):
    pass


class ReferenceStateful(Serializable):
    pass


def test_object_id_generates_nonce_and_trusted_decode_preserves_it():
    generated = ObjectId(("team", "trial"))
    decoded = ObjectId.from_data(generated.to_data())

    assert decoded == generated
    assert decoded.nonce == generated.nonce
    assert str(generated) == f"team/trial~{generated.nonce.hex[:12]}..."


def test_reference_identity_keeps_trajectories_and_ephemeral_graphs_distinct():
    first_ephemeral = ObjectRef(Definition(ReferenceEphemeral).concretize(), {})
    second_ephemeral = ObjectRef(Definition(ReferenceEphemeral).concretize(), {})
    assert first_ephemeral == second_ephemeral
    assert hash(first_ephemeral) == hash(second_ephemeral)

    definition = Definition(ReferenceStateful).concretize()
    first = ObjectRef(definition, {GraphPath(): ObjectId()})
    second = ObjectRef(definition.copy_graph(), {GraphPath(): ObjectId()})
    first_state = StateRef(first, {GraphPath(): "pkl-" + "a" * 64})
    second_state = StateRef(second, {GraphPath(): "pkl-" + "a" * 64})
    assert first != second
    assert first_state != second_state


@pytest.mark.parametrize("namespace", ["team", ("",), ("has space",), ("x" * 65,)])
def test_object_id_rejects_malformed_namespaces(namespace):
    with pytest.raises((TypeError, ValueError)):
        ObjectId(namespace)


def test_object_id_rejects_malformed_trusted_decode():
    with pytest.raises(ValueError):
        ObjectId.from_data({"namespace": ["team"], "nonce": "not-a-uuid"})

    with pytest.raises(ValueError):
        ObjectId.from_data({"namespace": ["team"], "nonce": str(UUID(int=0)), "extra": 1})


def test_reference_values_import_without_optional_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.core.reference_values; "
            "assert not ({'tensorflow', 'torch', 'jax'} & set(sys.modules))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
