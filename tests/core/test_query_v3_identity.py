import pytest

from dryml.core import Definition, ObjectId, ObjectRef, Serializable, StateRef
from dryml.core.definition import ConcreteDefinition
from dryml.core.query.identity import IdentitySet, Occurrence, OccurrenceSet, SourceEvidence
from dryml.core.utils.graph.path import GraphPath, Key


class V3IdentityLeaf(Serializable):
    def __init__(self, value):
        self.value = value


class V3IdentityPair(Serializable):
    def __init__(self, first, second):
        self.first = first
        self.second = second


def _graph_distinct_parents():
    shared = Definition(V3IdentityLeaf, "same").concretize()
    independent_first = Definition(V3IdentityLeaf, "same").concretize()
    independent_second = Definition(V3IdentityLeaf, "same").concretize()
    return (
        Definition(V3IdentityPair, shared, shared).concretize(),
        Definition(V3IdentityPair, independent_first, independent_second).concretize(),
    )


def test_identity_set_characterizes_graph_identity_without_changing_cdef_equality():
    shared_parent, independent_parent = _graph_distinct_parents()

    assert shared_parent == independent_parent
    assert not shared_parent.graph_equal(independent_parent)
    assert IdentitySet((shared_parent, independent_parent)).count() == 2


def test_identity_set_keeps_graph_distinct_members_in_one_forced_digest_bucket(monkeypatch):
    shared_parent, independent_parent = _graph_distinct_parents()
    monkeypatch.setattr(ConcreteDefinition, "graph_hash", lambda self: "0" * 64)

    values = IdentitySet((shared_parent, independent_parent))

    assert values.count() == 2
    assert set(values) == {shared_parent}


def test_ordering_avoids_extra_graph_encoding_without_digest_ties(monkeypatch):
    from dryml.core import cdef_codec

    first = Definition(V3IdentityLeaf, "first").concretize()
    second = Definition(V3IdentityLeaf, "second").concretize()
    values = IdentitySet((first, second))
    monkeypatch.setattr(
        cdef_codec, "cdef_graph_hash",
        lambda root: (_ for _ in ()).throw(AssertionError("extra graph encoding")),
    )

    assert len(tuple(values)) == 2


def test_reference_identity_uses_complete_object_and_state_reference_values():
    cdef = Definition(V3IdentityLeaf, "target").concretize()
    first = ObjectRef(cdef, {GraphPath(): ObjectId()})
    second = ObjectRef(cdef.copy_graph(), {GraphPath(): ObjectId()})
    first_state = StateRef(first, {GraphPath(): "pkl-" + "1" * 64})
    second_state = StateRef(second, {GraphPath(): "pkl-" + "1" * 64})

    assert IdentitySet((first, second)).count() == 2
    assert IdentitySet((first_state, second_state)).count() == 2


def test_identity_set_deduplicates_replicas_and_stays_detached_from_source():
    cdef = Definition(V3IdentityLeaf, "target").concretize()
    first_source = _ClosedSource()
    second_source = _ClosedSource()
    values = IdentitySet(
        (
            (cdef, SourceEvidence.from_source(first_source)),
            (cdef.copy_graph(), SourceEvidence.from_source(second_source)),
        )
    )
    first_source.close()
    second_source.close()

    assert values.count() == 1
    assert len(values.evidence_for(cdef).sources) == 2
    assert tuple(values) == (cdef,)


def test_occurrence_set_keeps_paths_distinct_and_refines_without_source_execution():
    owner = Definition(V3IdentityPair, None, None).concretize()
    target = Definition(V3IdentityLeaf, "same").concretize()
    occurrences = OccurrenceSet(
        (
            Occurrence(owner, GraphPath((Key("first"),)), target),
            Occurrence(owner, GraphPath((Key("second"),)), target),
        ),
        bounded=True,
    )
    calls = []
    refined = occurrences.query(
        lambda occurrence: calls.append(occurrence)
        or occurrence.path == GraphPath((Key("first"),))
    )

    with pytest.raises(TypeError, match="explicit terminal"):
        bool(refined)
    assert calls == []
    assert refined.collect().one().path == GraphPath((Key("first"),))
    assert refined.collect().bounded
    assert refined.exists()


def test_fixed_set_algebra_propagates_boundedness_and_diagnostics_are_sanitized():
    secret = "credential-like-private-value"
    first = IdentitySet(
        ((Definition(V3IdentityLeaf, secret).concretize(), SourceEvidence.from_source(secret)),),
        bounded=True,
    )
    second = IdentitySet((Definition(V3IdentityLeaf, "other").concretize(),))

    assert first.union(second).bounded
    assert first.intersection(first).bounded
    assert secret not in repr(first)
    assert secret not in repr(first.diagnostic())


def test_fixed_set_rejects_invalid_members_and_implicit_query_truth_testing():
    with pytest.raises(TypeError, match="identity members"):
        IdentitySet((object(),))
    with pytest.raises(TypeError, match="exact bool"):
        OccurrenceSet(bounded=1)
    with pytest.raises(TypeError, match="callable"):
        IdentitySet().query("not-a-predicate")


class _ClosedSource:
    def close(self):
        pass
