"""DefinitionRecord closure coverage for graphs with ephemeral nodes."""

import sys

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.artifacts import CachedDataset
from dryml.core import Definition, EdgePolicy, Object, Par, QueryDomainError, Repo, Serializable, StateRef, Template
from dryml.core.query import RelationshipKind
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_graph import EdgeKind
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict
from dryml.core.links import DefLink
from dryml.core.query.containment import (
    iter_containment_occurrences,
    iter_containment_owners,
    visit_containment_occurrences,
)
from dryml.core.query.model import ContainmentHop, DefinitionOccurrence, containment_witness_key
from dryml.core.quoted import QuotedDef
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord
from dryml.core.utils.graph.path import GraphPath
from dryml.data import Dataset


class QueryLeaf(Object):
    """Ephemeral child retained structurally by its enclosing definition."""

    def __init__(self, name):
        self.name = name


class QueryParent(Serializable):
    """Stateful root used to publish an enclosing definition closure."""

    def __init__(self, child, *, label="parent"):
        self.child = child
        self.label = label

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Publish no payload files for this structural test value."""


class QueryReferenceLeaf(Serializable):
    """Stateful reference target whose complete identities remain inspectable."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Publish no payload files for this exact-reference test value."""


class QueryDataset(Dataset):
    """Inert Dataset definition used to exercise CachedDataset reference retention."""

    def __iter__(self):
        return iter(())


class QueryRecipeOwner(Object):
    """Stored owner whose symbolic recipe is inert Template-role quotation data."""

    def __init__(self, recipe: Template):
        self.recipe = recipe


def test_symbolic_template_recipe_is_not_a_reference_containment_edge(tmp_path):
    """Quoted symbolic Definitions do not invent a traversable concrete target."""

    repo = Repo(DirStore(tmp_path / "store"))
    owner = QueryRecipeOwner(Definition(QueryLeaf, Par("name")), repo=repo)
    target = Definition(QueryLeaf, "child").concretize()
    repo.save_object(owner)

    assert isinstance(owner.definition.parameters["recipe"].target, QuotedDef)
    assert repo.query().cdefs().stored().nested(target, edges=EdgePolicy.ALL).owners().count() == 0
    assert repo.query().cdefs().stored().nested(
        target, edges=EdgePolicy(frozenset((RelationshipKind.REFERENCE,)))
    ).count() == 0


def test_cached_dataset_definition_only_reference_containment(tmp_path):
    """A retained CachedDataset source is discoverable without source execution."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = QueryDataset(repo=repo)
    cached = CachedDataset(source, repo=repo)

    repo.save_object(cached)

    assert source.definition not in tuple(repo.query().cdefs().stored().collect())
    assert repo.query().cdefs().stored().nested(
        source.definition, edges=EdgePolicy(frozenset((RelationshipKind.REFERENCE,)))
    ).owners().cdefs().one() == cached.definition


def test_save_records_definition_closure_for_ephemeral_child(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("child", repo=repo)
    parent = QueryParent(child, repo=repo)

    repo.save_object(parent)

    assert store.read_definition_record(DefinitionRecord(parent.definition).digest)
    assert store.read_definition_record(DefinitionRecord(child.definition).digest)


def test_repeated_ephemeral_child_has_one_definition_record(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("shared", repo=repo)
    parent = QueryParent([child, child], repo=repo)

    repo.save_object(parent)

    records = tuple(store.iter_definition_records())
    assert {record.definition for record in records} == {parent.definition, child.definition}


@pytest.mark.parametrize(
    ("path_kinds", "edges", "contains_ref", "matches"),
    (
        ((EdgeKind.MATERIALIZE,), "materialize", False, True),
        ((EdgeKind.MATERIALIZE,), "materialize", True, False),
        ((EdgeKind.MATERIALIZE,), "ref", False, False),
        ((EdgeKind.MATERIALIZE,), "all", False, True),
        ((EdgeKind.MATERIALIZE,), "all", True, False),
        ((EdgeKind.MATERIALIZE, EdgeKind.MATERIALIZE), "materialize", False, True),
        ((EdgeKind.MATERIALIZE, EdgeKind.MATERIALIZE), "ref", False, False),
        ((EdgeKind.MATERIALIZE, EdgeKind.MATERIALIZE), "all", False, True),
        ((EdgeKind.MATERIALIZE, EdgeKind.MATERIALIZE), "all", True, False),
        ((EdgeKind.REF,), "ref", False, True),
        ((EdgeKind.REF,), "ref", True, True),
        ((EdgeKind.REF,), "materialize", False, False),
        ((EdgeKind.REF,), "all", False, True),
        ((EdgeKind.REF,), "all", True, True),
        ((EdgeKind.REF, EdgeKind.REF), "ref", False, True),
        ((EdgeKind.REF, EdgeKind.REF), "materialize", False, False),
        ((EdgeKind.REF, EdgeKind.REF), "all", False, True),
        ((EdgeKind.REF, EdgeKind.REF), "all", True, True),
        ((EdgeKind.MATERIALIZE, EdgeKind.REF), "all", False, True),
        ((EdgeKind.MATERIALIZE, EdgeKind.REF), "materialize", False, False),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE), "all", False, True),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE), "materialize", False, False),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE, EdgeKind.REF), "all", True, True),
        ((EdgeKind.MATERIALIZE, EdgeKind.REF), "ref", False, False),
        ((EdgeKind.MATERIALIZE, EdgeKind.REF), "all", True, True),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE), "ref", False, False),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE), "all", True, True),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE, EdgeKind.REF), "materialize", False, False),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE, EdgeKind.REF), "ref", False, False),
        ((EdgeKind.REF, EdgeKind.MATERIALIZE, EdgeKind.REF), "all", False, True),
    ),
)
def test_containment_walker_honors_every_hop_policy(
    path_kinds, edges, contains_ref, matches,
):
    target = QueryLeaf("target").definition
    current = target
    for kind in reversed(path_kinds):
        value = current if kind is EdgeKind.MATERIALIZE else DefLink.finalized(kind, current)
        current = QueryParent(value).definition

    occurrences = tuple(
        iter_containment_occurrences(
            (current,), target, edges=edges, contains_ref=contains_ref,
        )
    )

    assert bool(occurrences) is matches
    if matches:
        assert tuple(hop.kind for hop in occurrences[0].hops) == path_kinds
        assert occurrences[0].target == target


def test_containment_walker_excludes_the_empty_root_path():
    root = QueryLeaf("root").definition

    assert tuple(iter_containment_occurrences((root,), root, edges="all")) == ()


def test_containment_reference_filter_is_path_local_and_preserves_dag_paths():
    target = QueryLeaf("target").definition
    unrelated = QueryLeaf("unrelated").definition
    material_branch = QueryParent(target).definition
    root = QueryParent(
        {
            "material": material_branch,
            "unrelated": DefLink.finalized(EdgeKind.REF, unrelated),
            "reference": DefLink.finalized(EdgeKind.REF, material_branch),
        }
    ).definition

    occurrences = tuple(
        iter_containment_occurrences(
            (root,), target, edges="all", contains_ref=True,
        )
    )

    assert len(occurrences) == 1
    assert tuple(hop.kind for hop in occurrences[0].hops) == (
        EdgeKind.REF, EdgeKind.MATERIALIZE,
    )


def test_containment_walker_keeps_shared_and_equal_private_node_paths_distinct():
    target = QueryLeaf("target").definition
    shared = QueryParent(target).definition
    equal_first = QueryLeaf("equal").definition
    equal_second = QueryLeaf("equal").definition
    root = QueryParent([shared, shared, equal_first, equal_second]).definition

    shared_occurrences = tuple(
        iter_containment_occurrences((root,), target, edges="materialize")
    )
    equal_occurrences = tuple(
        iter_containment_occurrences(
            (root,), equal_first, edges="materialize",
        )
    )

    assert len(shared_occurrences) == 2
    assert len({item.path for item in shared_occurrences}) == 2
    assert len(equal_occurrences) == 2
    assert len({item.path for item in equal_occurrences}) == 2


def test_containment_owner_projection_is_existential_and_callback_reuses_witnesses(
    monkeypatch,
):
    target = QueryLeaf("target").definition
    shared = QueryParent(target).definition
    root = QueryParent([shared, shared]).definition
    captured = []

    def fail_path_enumeration(*args, **kwargs):
        raise AssertionError("owner projection enumerated raw occurrences")

    with monkeypatch.context() as patch:
        patch.setattr(
            "dryml.core.query.containment._iter_root_occurrences",
            fail_path_enumeration,
        )
        assert tuple(iter_containment_owners((root,), target)) == (root,)

    visit_containment_occurrences(
        (root,), target, captured.append, edges="materialize",
    )
    assert len(captured) == 2


def test_containment_hop_evidence_preserves_legacy_definition_occurrence_identity():
    definition = QueryLeaf("target").definition
    original = DefinitionOccurrence(definition, GraphPath(), definition)
    enriched = DefinitionOccurrence(
        definition,
        GraphPath(),
        definition,
        (ContainmentHop(GraphPath(), EdgeKind.REF),),
    )

    assert enriched == original
    assert hash(enriched) == hash(original)
    assert enriched.target == definition
    assert containment_witness_key(enriched) != containment_witness_key(original)


def test_containment_walker_matches_terminal_exact_references_without_imports(
    tmp_path, monkeypatch,
):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(QueryReferenceLeaf("target", repo=repo))
    other_state = StateRef(
        state.object,
        {path: "pkl-" + "0" * 64 for path in state.states},
    )
    root = ConcreteDefinition._from_persisted_record(
        QueryParent,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments(((
            "child",
            FrozenDict({
                "bare": state.object,
                "ref": DefLink.finalized(EdgeKind.REF, state),
            }),
        ),)),
    )
    def fail_resolution(*args, **kwargs):
        raise AssertionError("containment traversal resolved a retained reference")

    monkeypatch.setattr(repo, "build_object_ref", fail_resolution)
    monkeypatch.setattr(repo, "load_state_ref", fail_resolution)
    before = set(sys.modules)

    object_occurrences = tuple(
        iter_containment_occurrences((root,), state.object, edges="materialize")
    )
    state_occurrences = tuple(
        iter_containment_occurrences((root,), state, edges="ref", contains_ref=True)
    )
    mismatched_states = tuple(
        iter_containment_occurrences((root,), other_state, edges="ref")
    )

    assert len(object_occurrences) == 1
    assert object_occurrences[0].target == state.object
    assert tuple(hop.kind for hop in object_occurrences[0].hops) == (EdgeKind.MATERIALIZE,)
    assert len(state_occurrences) == 1
    assert state_occurrences[0].target == state
    assert tuple(hop.kind for hop in state_occurrences[0].hops) == (EdgeKind.REF,)
    assert mismatched_states == ()
    assert set(sys.modules) == before
