"""Focused U2 contracts for private Query V3 source and authority capture."""

from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, ObjectId, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.core.query.identity import IdentitySet
from dryml.core.query.relationships import RelationshipKind, iter_direct_relationships
from dryml.core.query.source import (
    RepoSource, SourceCapture, StoreSource,
)
from dryml.core.links import DefLink
from dryml.core.cdef_graph import EdgeKind


class SourceLeaf(Serializable):
    """Small stateful graph member used to observe captured identity inventory."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "name").write_text(self.name, encoding="ascii")

    def restore_state_from_dir_imp(self, directory, *, codec):
        self.name = Path(directory, "name").read_text(encoding="ascii")


class SourcePair(Object):
    """Root fixture that owns one child without requiring payload inspection."""

    def __init__(self, child):
        super().__init__()
        self.child = child


class AliasBlindDirStore(DirStore):
    """Backend fixture whose direct alias reads do not prove broad enumeration."""

    def _query_v3_inventory_capabilities(self):
        return replace(super()._query_v3_inventory_capabilities(), object_aliases="unsupported")


def test_store_capture_includes_represented_references_and_owned_derivatives(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    root = SourcePair(SourceLeaf("child", repo=repo), repo=repo)
    state = repo.save_object(root)
    child_path = GraphPath((Parameter("child"),))
    child = state.at(child_path).object_view()

    monkeypatch.setattr(
        store, "_authenticate_state_data",
        lambda *_args, **_kwargs: pytest.fail("source inventory opened state payload bytes"),
    )
    facts = SourceCapture().capture_store(StoreSource(store))
    known = facts.knowledge()

    assert state in known
    assert state.object_view() in known
    assert state.creation_view() in known
    assert child in known
    assert child.creation_view() in known
    assert not facts.is_stored(state.at(child_path))
    assert facts.holds_metadata(child)


def test_stored_membership_uses_distinct_root_object_and_state_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    standalone = Definition(SourceLeaf, "root").concretize()
    store.publish_node(standalone)
    root = SourcePair(SourceLeaf("child", repo=repo), repo=repo)
    state = repo.save_object(root)
    child = state.at(GraphPath((Parameter("child"),)))
    facts = SourceCapture().capture_store(store)

    assert facts.is_stored(standalone)
    assert facts.is_stored(state.object_view())
    assert facts.is_stored(state)
    assert not facts.is_stored(child.creation_view())
    assert not facts.is_stored(child)
    assert facts.holds_metadata(child)


def test_repo_capture_adds_retained_cache_receipts_without_new_capture(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    strong_only = SourceLeaf("strong-only", repo=repo)
    weak_only = SourceLeaf("weak-only", repo=repo)
    saved = SourceLeaf("cached", repo=repo)
    repo.cache_strong(strong_only)
    repo.cache_weak(weak_only)
    receipt = repo.save_object(saved)
    repo.cache_strong(saved)

    monkeypatch.setattr(repo, "save_object", lambda *_args, **_kwargs: pytest.fail("query capture saved"))
    strong = SourceCapture().capture_repo(RepoSource(repo, weak=False))
    known = SourceCapture().capture_repo(RepoSource(repo, weak=True))

    assert strong_only.definition in strong
    assert strong.query().object_id(strong_only.object_id).object_refs().exists()
    assert weak_only.definition not in strong
    assert weak_only.definition in known
    assert receipt in known


def test_repo_query_strong_cache_selection_preserves_store_knowledge(tmp_path):
    """weak=False excludes weak-only facts without narrowing Store knowledge."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    stored = repo.save_object(SourceLeaf("stored", repo=repo))
    strong = SourceLeaf("strong", repo=repo)
    weak = SourceLeaf("weak", repo=repo)
    repo.cache_strong(strong)
    repo.cache_weak(weak)
    query = repo.query(weak=False).cdefs()

    assert query.sel(stored.creation_view()).one().graph_equal(stored.creation_view())
    assert query.sel(strong.definition).one().graph_equal(strong.definition)
    assert query.sel(weak.definition).count() == 0
    with pytest.raises(TypeError, match="exact bool"):
        repo.query(weak=1)


def test_fixed_collections_remain_explicit_members_without_source_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = repo.save_object(SourceLeaf("saved", repo=repo))
    extra = Definition(SourceLeaf, "extra").concretize()
    fixed = IdentitySet((extra,))

    assert tuple(fixed) == (extra,)
    assert saved not in fixed
    assert tuple(fixed.query().collect()) == (extra,)


def test_unsupported_alias_enumeration_fails_preflight_instead_of_reading_partial_inventory(tmp_path):
    store = AliasBlindDirStore(tmp_path / "store")

    # Retired alias enumeration cannot constrain current named membership reads.
    from dryml.core.store.records import NodeNameRecord
    root = store.publish_node(Definition(SourceLeaf, "named").concretize())
    store.compare_and_set_node_name(NodeNameRecord("chosen", "recipe", root.digest), expected=None)
    assert store.query().cdefs().alias("chosen").one().parameters["name"] == "named"


def test_broad_capture_reuses_one_record_family_pass_and_exact_state_reads_directly(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    state = repo.save_object(SourceLeaf("saved", repo=repo))
    from dryml.core.query import authority
    calls = []
    original = authority.capture_store_projection
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(authority, "capture_store_projection", counted)
    base = store.query()
    base.union(base).collect()

    assert calls == [1]
    monkeypatch.setattr(authority, "capture_store_projection", lambda *a, **kw: pytest.fail("exact read enumerated inventory"))
    assert store.query().local_states().state_key(state.state.key).one().key == state.state.key


def test_stored_restriction_builds_root_membership_once_per_cut(tmp_path, monkeypatch):
    from dryml.core.query.source import _NodeFacts
    from dryml.core.query.query import IdentityQuery

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(SourceLeaf("first", repo=repo))
    repo.save_object(SourceLeaf("second", repo=repo))
    original = _NodeFacts._authoritative_root_definitions
    calls = []

    def counted(self):
        calls.append(1)
        yield from original(self)

    monkeypatch.setattr(_NodeFacts, "_authoritative_root_definitions", counted)
    assert IdentityQuery.from_store(store).stored().saved().count() == 2
    assert len(calls) <= 1


def test_distinct_zip_transactions_and_direct_relationships_remain_independent(tmp_path):
    archive = tmp_path / "store.zip"
    cdef = Definition(
        SourcePair, Definition(SourceLeaf, "edge").concretize(),
    ).concretize()
    seed = ZipStore(archive)
    seed.publish_node(cdef)
    seed.commit()
    seed.close()
    first = ZipStore.open_existing(archive)
    second = ZipStore.open_existing(archive)
    try:
        first_facts, second_facts = SourceCapture().capture_stores((first, second))
        assert first_facts is not second_facts

        edges = tuple(iter_direct_relationships(cdef))
        assert [edge.kind for edge in edges] == [RelationshipKind.MATERIALIZE]
    finally:
        first.close()
        second.close()


def test_object_reference_authority_conflicts_fail_within_and_across_stores(tmp_path):
    """One ObjectId cannot authoritatively identify incompatible ObjectRefs."""

    object_id = ObjectId(("conflict",))
    first_ref = Definition(SourceLeaf, "first").concretize().with_identity(object_id)
    second_ref = Definition(SourceLeaf, "second").concretize().with_identity(object_id)
    combined = DirStore(tmp_path / "combined")
    combined.publish_node(first_ref)
    with pytest.raises(Exception, match="projection conflicts"):
        combined.publish_node(second_ref)

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first.publish_node(first_ref)
    second.publish_node(second_ref)
    for stores in ((first, second), (second, first)):
        with pytest.raises(Exception, match="object-reference authority conflicts"):
            Repo(stores).query().object_refs().collect()


def test_embedded_object_references_do_not_become_independent_authority(tmp_path):
    """Structural references may conflict without becoming declaration authority."""

    object_id = ObjectId(("embedded",))
    first_ref = Definition(SourceLeaf, "first").concretize().with_identity(object_id)
    second_ref = Definition(SourceLeaf, "second").concretize().with_identity(object_id)
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first.publish_node(Definition(SourcePair, DefLink.finalized(EdgeKind.REF, first_ref)).concretize())
    second.publish_node(Definition(SourcePair, DefLink.finalized(EdgeKind.REF, second_ref)).concretize())
    # Identity-contributing Ref publications now require consistent anchors too.
    from dryml.core.query.model import QueryError
    with pytest.raises(QueryError, match="authority conflicts"):
        Repo((first, second)).query().object_refs().count()


def test_exact_embedded_state_and_owned_projection_preserve_full_knowledge(tmp_path):
    """Exact live selection keeps embedded projections without claiming storage."""

    writer = Repo(DirStore(tmp_path / "writer"))
    root = SourcePair(SourceLeaf("child", repo=writer), repo=writer)
    state = writer.save_object(root)
    child = state.at(GraphPath((Parameter("child"),)))
    embedded = DirStore(tmp_path / "embedded", query_index="memory")
    embedded.publish_node(Definition(SourcePair, DefLink.finalized(EdgeKind.REF, state)).concretize())
    reopened = Repo(DirStore.open_existing(embedded.base_dir, query_index="memory"))
    cached = Repo(DirStore(tmp_path / "cache-only", query_index="memory"))
    cached.cache_strong(root)
    for query in (embedded.query(), reopened.query(), cached.query(),
                  Repo((embedded, writer.stores[0])).query()):
        fixed = query.collect()
        for target in (state, child):
            expected = fixed.query().sel(target).collect()
            live = query.sel(target).collect()
            assert len(live) == len(expected) == 1
            assert live.one().graph_equal(target) and expected.one().graph_equal(target)
            assert live.sources(live.one()) == expected.sources(expected.one())
            assert query.sel(target).exists()
        assert query.sel(child).stored().count() == 0
    assert embedded.query().sel(state).stored().count() == 0
    assert writer.stores[0].query().sel(state).stored().count() == 1


def test_exact_state_scoped_scalars_validate_complete_reference_authority(tmp_path):
    """A direct hit cannot suppress a scoped selected-root conflict."""

    from dryml.core.query.model import QueryError

    direct = DirStore(tmp_path / "direct", query_index="memory")
    repo = Repo(direct)
    state = repo.save_object(SourceLeaf("saved", repo=repo))
    conflicting = DirStore(tmp_path / "conflicting", query_index="memory")
    conflicting.publish_node(Definition(SourceLeaf, "other").concretize().with_identity(state.object_id))
    query = Repo((direct, conflicting)).query().sel(state)
    with pytest.raises(QueryError, match="authority conflicts"):
        query.exists()
    for scoped in (query.stored(), query.in_source(direct), query.alias("absent")):
        for terminal in (scoped.exists, scoped.one_or_none, scoped.count, scoped.collect):
            with pytest.raises(QueryError, match="authority conflicts"):
                terminal()


def test_exact_state_negative_requires_complete_knowledge_or_scan_refusal(tmp_path, monkeypatch):
    """Neither an embedded direct miss nor an empty Store justifies a no-scan negative."""

    from dryml.core.query.model import QueryWouldScanError

    writer = Repo(DirStore(tmp_path / "writer", query_index="memory"))
    state = writer.save_object(SourceLeaf("saved", repo=writer))
    embedded = DirStore(tmp_path / "embedded", query_index="memory")
    embedded.publish_node(Definition(SourcePair, DefLink.finalized(EdgeKind.REF, state)).concretize())
    empty = DirStore(tmp_path / "empty", query_index="memory")
    assert embedded.query().sel(state).exists()
    assert not empty.query().sel(state).exists()
    for store in (embedded, empty):
        monkeypatch.setattr("dryml.core.query.authority.capture_store_projection", lambda *a, **kw: pytest.fail("forbidden fallback scanned"))
        with pytest.raises(QueryWouldScanError, match="inventory scan"):
            store.query().sel(state).scan_policy("forbid").exists()


def test_exact_direct_state_collect_completes_embedded_source_evidence(tmp_path):
    """Direct authority proves membership, not the complete contributor set."""

    direct = DirStore(tmp_path / "direct", query_index="memory")
    writer = Repo(direct)
    state = writer.save_object(SourceLeaf("saved", repo=writer))
    embedded = DirStore(tmp_path / "embedded", query_index="memory")
    embedded.publish_node(Definition(SourcePair, DefLink.finalized(EdgeKind.REF, state)).concretize())
    query = Repo((direct, embedded)).query()
    expected = query.collect().query().sel(state).collect()
    result = query.sel(state).collect()
    assert result.sources(state) == expected.sources(state)
    assert len(result.sources(state)) == 2
    assert query.sel(state).in_source(embedded).one().graph_equal(state)


def test_selected_combinations_and_replacement_tokens_do_not_hide_conflicts(tmp_path, monkeypatch):
    """Proof reuse is domain/token-local, including unchanged demand recapture."""

    from dryml.core.query import source as source_module
    from dryml.core.query.model import QueryError

    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    third = DirStore(tmp_path / "third", query_index="memory")
    identity = ObjectId()
    ref = Definition(SourceLeaf, "first").concretize().with_identity(identity)
    conflict = Definition(SourceLeaf, "conflict").concretize().with_identity(identity)
    first.publish_node(ref)
    third.publish_node(conflict)
    capture = SourceCapture()
    passes = []
    original = source_module.validate_captured_reference_authority

    def validate(facts):
        passes.append(tuple(item.capture_token for item in facts))
        return original(facts)

    monkeypatch.setattr(source_module, "validate_captured_reference_authority", validate)
    initial = capture.capture_stores((first, second))
    duplicate = DirStore.open_existing(first.base_dir, query_index="memory")
    assert capture.capture_stores((second, duplicate, first)) == tuple(reversed(initial))
    assert len(passes) == 1
    enriched = capture.capture_stores((first, second), metadata_scopes=frozenset(("object",)))
    assert all(old.capture_token is not new.capture_token for old, new in zip(initial, enriched))
    assert len(passes) == 2
    with pytest.raises(QueryError, match="authority conflicts"):
        capture.capture_stores((first, third))
    second.publish_node(conflict)
    capture.active = True
    with pytest.raises(QueryError, match="changed"):
        capture.capture_store(second, metadata_scopes=frozenset(("object", "lineage")))
    with pytest.raises(QueryError, match="authority conflicts"):
        SourceCapture().capture_stores((first, second))


def test_repo_knowledge_reuse_separates_cache_tiers_and_retained_lookup(tmp_path):
    """Shared Store cuts never alias full knowledge with root-or-cache lookup."""

    from dryml.core.query.source import RetainedRepoLookupSource

    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    strong = SourceLeaf("strong", repo=repo)
    weak = SourceLeaf("weak", repo=repo)
    repo.cache_strong(strong)
    repo.cache_weak(weak)
    embedded = Definition(SourceLeaf, "embedded").concretize()
    store.publish_node(Definition(SourcePair, embedded).concretize())
    capture = SourceCapture()
    strong_members = capture.capture_repo(RepoSource(repo, weak=False))
    all_members = capture.capture_repo(RepoSource(repo, weak=True))
    retained = capture.capture_retained_lookup_repo(RetainedRepoLookupSource(repo, weak=False))
    assert embedded in strong_members and embedded in all_members
    assert embedded not in retained
    assert weak.definition not in strong_members and weak.definition in all_members
    assert weak.definition not in retained and strong.definition in retained
    assert capture.capture_rounds == 1
    other_repo = Repo(store)
    assert strong.definition not in capture.capture_repo(RepoSource(other_repo))
    intersection = repo.query(weak=False).intersection(repo.query(weak=True)).collect()
    assert weak.definition not in intersection and strong.definition in intersection
    assert weak.definition in repo.query(weak=False).union(repo.query(weak=True)).collect()


def test_metadata_replacement_has_fresh_failure_facts_and_invalidates_results(tmp_path, monkeypatch):
    """Unchanged identity authority cannot preserve an earlier metadata proof."""

    from dryml.core.query import field
    from dryml.core.query.model import QueryError
    from dryml.core.query.source import _MetadataReadFailure, _SourceRecapture

    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    state = repo.save_object(SourceLeaf("saved", repo=repo))
    capture = SourceCapture()
    old = capture.capture_store(store, metadata_scopes=frozenset(("object",)))
    old.knowledge()
    capture.active = True
    monkeypatch.setattr(store, "read_metadata", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("private")))
    with pytest.raises(_SourceRecapture) as changed:
        capture.capture_store(store, metadata_scopes=frozenset(("object", "lineage")))
    assert changed.value.invalidated
    new = capture.capture_store(store)
    assert new.capture_token is not old.capture_token
    assert isinstance(new.captured_metadata(state.object_view(), "object"), _MetadataReadFailure)
    assert new.knowledge()._entries.keys() == old.knowledge()._entries.keys()
    query = store.query().sel(state).where(field("object", "missing").missing())
    with pytest.raises(QueryError, match="metadata read failed"):
        query._collect_with_capture(capture)


def test_zip_inventory_preserves_uncommitted_transaction_domains(tmp_path):
    """Same archive handles retain independent extracted authority and tokens."""

    archive = tmp_path / "transactions.zip"
    seed = ZipStore(archive)
    base = Definition(SourceLeaf, "base").concretize()
    added = Definition(SourceLeaf, "added").concretize()
    seed.publish_node(base)
    seed.commit()
    seed.close()
    first, second = ZipStore.open_existing(archive), ZipStore.open_existing(archive)
    try:
        first.publish_node(added)
        capture = SourceCapture()
        one, two = capture.capture_stores((first, second, first))
        assert one.capture_token is not two.capture_token
        assert capture.source_cuts == 2
        assert added in one.knowledge() and added not in two.knowledge()
        assert one.source != two.source
    finally:
        first.close()
        second.close()
    with pytest.raises(RuntimeError, match="closed"):
        SourceCapture().capture_store(first)
