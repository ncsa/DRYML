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


def test_local_state_exact_values_and_no_alternative_bindings():
    from dryml.core import LocalState, StateKey

    definition = Definition(ReferenceStateful).concretize()
    key = StateKey("a" * 64, "pkl-" + "b" * 64)
    state = LocalState(key, definition)
    same = LocalState(key, Definition(ReferenceStateful).concretize())
    assert definition.with_state(state).with_state(same).state is state
    assert state.codec == "pkl"
    with pytest.raises(ValueError):
        StateKey("not-a-digest", key.codec_hash)
    with pytest.raises(ValueError):
        StateKey(key.cdef_digest, "bare-codec-hash")
    with pytest.raises(TypeError):
        LocalState(key, Definition(ReferenceStateful))
    with pytest.raises(TypeError, match="LocalState"):
        definition.with_state(key)
    with pytest.raises(TypeError, match="LocalState"):
        definition.with_state({state, same})


def test_saved_subtree_maps_are_rebased_and_exclude_reference_ownership(monkeypatch):
    from dryml.core import IdentitySpec, LocalState, StateKey, StateSpec, require_saved
    from dryml.core.cdef_graph import EdgeKind
    from dryml.core.links import DefLink
    from dryml.core.repo import Repo

    child = Definition(ReferenceStateful).concretize()
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), child)
    child = child.with_identity(ObjectId()).with_state(state)
    root = Definition(children=[child, child], ref=child.ref(), quoted=child.quote())
    # Pure metadata must not enter any existing materialization/publication seam.
    def forbidden(*args, **kwargs):
        pytest.fail("metadata performed runtime or payload IO")
    monkeypatch.setattr(Repo, "load_or_build", forbidden)
    monkeypatch.setattr(Repo, "load_state_ref", forbidden)
    monkeypatch.setattr(Repo, "save", forbidden)
    monkeypatch.setattr(ObjectId, "__init__", forbidden)
    monkeypatch.setattr(ReferenceStateful, "__init__", forbidden)
    monkeypatch.setattr("dryml.core.symbol.ImportRef.resolve", forbidden)
    assert len(root.identity_spec()) == len(root.state_spec()) == 1
    assert isinstance(root.identity_spec(), IdentitySpec)
    assert isinstance(root.state_spec(), StateSpec)
    selected = root.at("$.children[1]")
    assert selected is child and require_saved(selected) is child
    assert dict(selected.state_spec()) == {GraphPath(): state}
    assert selected.object_id == child.object_id
    assert root.at("$.ref") is child and root.at("$.quoted") is child
    for operation in (root.identity_spec, root.state_spec):
        assert all("ref" not in str(path) and "quoted" not in str(path) for path in operation())
    with pytest.raises(AttributeError):
        root.state_spec()._root = child
