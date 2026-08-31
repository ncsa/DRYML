import pytest

from dryml.core import Definition, Mat, ObjectId, ObjectRef, Ref, StateRef
from dryml.core.cdef_codec import CDefGraphCodecError, decode_cdef_graph, encode_cdef_graph
from dryml.core.object import Object, Serializable
from dryml.core.utils.graph.path import GraphPath


class CanonicalLeaf(Serializable):
    pass


class CanonicalOwner(Object):
    def __init__(self, value):
        self.value = value


def _state_ref():
    leaf = Definition(CanonicalLeaf).concretize()
    object_ref = ObjectRef(leaf, {GraphPath(): ObjectId(("canonical",))})
    return StateRef(object_ref, {GraphPath(): "pkl-" + "b" * 64})


def test_exact_references_are_atomic_canonical_values_and_codec_round_trip():
    state = _state_ref()
    cdef = Definition(CanonicalOwner, {"state": state, "ref": Ref(state)}).concretize()

    assert cdef.parameters["value"]["state"] is state
    assert cdef.parameters["value"]["ref"].target is state
    decoded = type(state).from_data(state.to_data())
    assert decoded == state
    assert decoded.definition.graph_equal(state.definition)
    assert decoded.definition is not state.definition


def test_soft_state_selector_resolves_once_before_cdef_identity():
    state = _state_ref()
    selector = state.object.state("best")

    class FakeRepo:
        calls = 0

        def resolve_state_selector(self, value):
            self.calls += 1
            assert value == selector
            return state

    repo = FakeRepo()
    cdef = Definition(CanonicalOwner, [selector, Ref(selector), Mat(selector)]).concretize(repo=repo)
    assert repo.calls == 1
    assert cdef.parameters["value"][0] is state
    assert cdef.parameters["value"][1].target is state
    assert cdef.parameters["value"][2] is state


def test_soft_state_selector_fails_without_or_outside_repo_scope():
    state = _state_ref()
    selector = state.object.state("best")
    with pytest.raises(KeyError, match="no alias"):
        Definition(CanonicalOwner, selector).concretize()

    class BadRepo:
        def resolve_state_selector(self, value):
            leaf = Definition(CanonicalLeaf).concretize()
            other = ObjectRef(leaf, {GraphPath(): ObjectId()})
            return StateRef(other, {GraphPath(): "pkl-" + "c" * 64})

    with pytest.raises(ValueError, match="outside its ObjectRef scope"):
        Definition(CanonicalOwner, selector).concretize(repo=BadRepo())


def test_soft_state_selector_is_a_stable_definition_and_selector_leaf():
    selector = _state_ref().object.state("best")
    definition = Definition(CanonicalOwner, selector)

    assert definition.stable_hash()
    assert definition.as_selector().root.parameters["value"] is selector


def test_cdef_graph_codec_rejects_state_selector_authority():
    cdef = Definition(CanonicalOwner, "value").concretize()
    authority = encode_cdef_graph(cdef)
    authority["nodes"][0]["parameters"]["items"][0][1] = {
        "kind": "state_selector_ref",
        "object": _state_ref().object.to_data(),
        "alias": "best",
    }

    with pytest.raises(CDefGraphCodecError, match="Unknown CDef graph value kind"):
        decode_cdef_graph(authority)
