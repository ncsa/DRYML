import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, ObjectRef, QueryDomainError, Repo, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DeclarationRecord, DefinitionRecord
from dryml.core.utils.graph.path import GraphPath, Parameter


class ReferenceQueryLeaf(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class ReferenceQueryWrapper(Object):
    def __init__(self, child):
        self.child = child


class ReferenceQueryRefParent(Object):
    def __init__(self, selected):
        self.selected = selected


def test_reference_filters_scan_authority_without_materializing(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))

    assert repo.references().object_id(state.object_id).object_refs().one() == state.object
    assert repo.references().namespace(state.object_id.namespace).object_refs().one() == state.object
    assert repo.references().definition(state.definition).object_refs().one() == state.object
    assert repo.references().state_hash(next(iter(state.states.values()))).state_refs().one() == state


def test_u4_prepared_selector_composes_with_reference_sidecars(tmp_path):
    """Prepared semantic constraints remain sound through reference sidecar scans."""
    from dryml.core import categorical_definition

    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    repo.save_object(ReferenceQueryLeaf(4, repo=repo))
    selector = categorical_definition(Definition(ReferenceQueryLeaf, 3))

    assert repo.query(selector).references().object_refs().one() == state.object


def test_object_id_lookup_is_closed_but_reference_query_returns_aggregate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    child = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    aggregate = repo.save_object(ReferenceQueryWrapper(child, repo=repo))

    assert repo.lookup_object_ref(child.object_id) == child.object
    assert aggregate.object in repo.references().object_id(child.object_id).object_refs()


def test_reference_filters_keep_exact_paths_aliases_and_all_ephemeral_refs(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    child = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    aggregate = repo.save_object(ReferenceQueryWrapper(child, repo=repo))
    repo.set_alias("aggregate", aggregate)

    path = GraphPath((Parameter("child"),))
    assert list(repo.references().contains(child.object).object_refs()) == [aggregate.object]
    assert list(repo.references().alias("aggregate").object_refs()) == [aggregate.object]
    assert list(repo.references().path(path).state_refs()) == [child]

    ephemeral = repo.save_object(ReferenceQueryWrapper("value", repo=repo))
    assert ephemeral.object.objects == {}
    assert repo.references().exact(ephemeral.object).object_refs().one() == ephemeral.object
    assert repo.references().definition(ephemeral.definition).state_refs().one() == ephemeral


def test_object_terminal_preserves_nested_ref_state_reference(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    parent = repo.save_object(
        ReferenceQueryRefParent(DefLink.finalized(EdgeKind.REF, state), repo=repo)
    )

    loaded = repo.query(parent.definition).stored().objects(cache="none").one()

    assert loaded.selected == state
    assert loaded.definition.parameters["selected"].target == state


def test_derived_candidates_cannot_hide_conflicting_store_authority(
    tmp_path, monkeypatch
):
    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="sqlite")
    repo = Repo([first, second])
    state = repo.save_object(ReferenceQueryLeaf(1, repo=repo), store=first)
    incompatible = ObjectRef(
        ReferenceQueryLeaf(2).definition,
        {"$": state.object_id},
    )
    second.write_definition_record(
        DefinitionRecord(incompatible.definition), stored_root=False
    )
    second.write_declaration_record(DeclarationRecord(incompatible))
    query = repo.references().object_id(state.object_id)

    monkeypatch.setattr(
        type(query),
        "_candidate_sources",
        lambda self, store: (
            (("state-ref", state.digest()),) if store is first else ()
        ),
    )

    with pytest.raises(RepoLoadError, match="incompatible closed-subtree authority"):
        query.object_refs()


def test_exact_reference_containment_entry_is_lazy_and_preserves_intent(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))

    monkeypatch.setattr(
        store,
        "iter_definition_records",
        lambda: (_ for _ in ()).throw(AssertionError("containment builder scanned Store")),
    )

    query = repo.query(state).nested(edges="all", contains_ref=True, refresh=False)

    assert query.selector is None
    assert query.containment_target == state
    assert query.containment_edges == "all"
    assert query.contains_ref is True
    assert query.refresh_policy is False
    object_query = repo.query(state.object).nested(edges="ref")
    assert object_query.containment_target == state.object
    assert object_query.containment_edges == "ref"


def test_exact_reference_containment_rejects_wrong_domains_projections_and_conversions(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))

    for domain in ("stored", "cached", "known"):
        with pytest.raises(QueryDomainError, match="[Ee]xact-reference"):
            getattr(repo.query(state), domain)()
    with pytest.raises(QueryDomainError, match="[Ee]xact-reference"):
        repo.query(state).categorical()
    with pytest.raises(QueryDomainError, match="reference authority"):
        repo.query(state).references()
    with pytest.raises(QueryDomainError, match="StateRef"):
        repo.query(state).nested().definitions()
    with pytest.raises(QueryDomainError, match="StateRef"):
        repo.query(state).nested().object_refs()


def test_reference_containing_adapts_unfiltered_source_and_source_scope(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))

    direct = repo.query(state).nested(edges="ref", contains_ref=True, refresh=False).in_store(store)
    adapted = repo.references().in_store(store).containing(
        state, edges="ref", contains_ref=True, refresh=False,
    )

    assert adapted == direct
    with pytest.raises(QueryDomainError, match="unfiltered"):
        repo.references().exact(state.object).containing(state)


def test_reference_containing_rejects_invalid_target_and_remains_nonexecuting(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))

    with pytest.raises(TypeError, match="containment target"):
        repo.references().containing(object())
    with pytest.raises(QueryDomainError, match="[Ee]xact-reference containment"):
        repo.query(state).nested().state_refs()

    assert repo.references().exact(state.object).object_refs().one() == state.object
