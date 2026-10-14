"""Focused U2 contracts for private Query V3 source and authority capture."""

from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, ObjectId, ObjectRef, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DeclarationRecord, DefinitionRecord
from dryml.core.store.zip import ZipStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.core.query.authority import _derive_identities
from dryml.core.query.identity import IdentitySet
from dryml.core.query.relationships import RelationshipKind, iter_direct_relationships
from dryml.core.query.source import (
    InventoryCapabilityError, RepoSource, SourceCapture, StoreSource,
)


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
    child = state.object.at(child_path)

    monkeypatch.setattr(
        store, "_validate_local_state_dir",
        lambda *_args, **_kwargs: pytest.fail("source inventory opened state payload bytes"),
    )
    facts = SourceCapture().capture_store(StoreSource(store))
    known = facts.knowledge()

    assert state in known
    assert state.object in known
    assert state.definition in known
    assert child in known
    assert child.definition in known
    assert not facts.is_stored(child)
    assert facts.holds_metadata(child)


def test_stored_membership_uses_distinct_root_object_and_state_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    standalone = Definition(SourceLeaf, "root").concretize()
    store.write_definition_record(DefinitionRecord(standalone))
    root = SourcePair(SourceLeaf("child", repo=repo), repo=repo)
    state = repo.save_object(root)
    child = state.object.at(GraphPath((Parameter("child"),)))
    facts = SourceCapture().capture_store(store)

    assert facts.is_stored(standalone)
    assert facts.is_stored(state.object)
    assert facts.is_stored(state)
    assert not facts.is_stored(child.definition)
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
    assert strong_only.object_ref in strong
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

    assert query.sel(stored.definition).one() == stored.definition
    assert query.sel(strong.definition).one() == strong.definition
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

    with pytest.raises(InventoryCapabilityError, match="complete Query V3 inventory"):
        SourceCapture().capture_store(store)


def test_broad_capture_reuses_one_record_family_pass_and_exact_state_reads_directly(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    state = repo.save_object(SourceLeaf("saved", repo=repo))
    calls = {}
    names = (
        "iter_definition_records",
        "iter_declaration_records", "iter_state_ref_records",
        "iter_object_alias_records", "iter_state_alias_records",
    )
    for name in names:
        original = getattr(store, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, counted)
    base = store.query()
    base.union(base).collect()

    assert calls == {name: 1 for name in names}
    for name in names:
        monkeypatch.setattr(store, name, lambda: pytest.fail("exact read enumerated inventory"))
    assert SourceCapture().read_exact_state(store, state) == state


def test_stored_restriction_builds_root_membership_once_per_cut(tmp_path, monkeypatch):
    from dryml.core.query.authority import CapturedStoreFacts
    from dryml.core.query.query import IdentityQuery

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(SourceLeaf("first", repo=repo))
    repo.save_object(SourceLeaf("second", repo=repo))
    original = CapturedStoreFacts._authoritative_root_definitions
    calls = []

    def counted(self):
        calls.append(1)
        yield from original(self)

    monkeypatch.setattr(CapturedStoreFacts, "_authoritative_root_definitions", counted)
    assert IdentityQuery.from_store(store).stored().cdefs().count() == 2
    assert len(calls) <= 1


def test_distinct_zip_transactions_and_direct_relationships_remain_independent(tmp_path):
    archive = tmp_path / "store.zip"
    cdef = Definition(
        SourcePair, Definition(SourceLeaf, "edge").concretize(),
    ).concretize()
    seed = ZipStore(archive)
    seed.write_definition_record(DefinitionRecord(cdef))
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
    first_ref = ObjectRef(
        Definition(SourceLeaf, "first").concretize(),
        {GraphPath(): object_id},
    )
    second_ref = ObjectRef(
        Definition(SourceLeaf, "second").concretize(),
        {GraphPath(): object_id},
    )
    combined = DirStore(tmp_path / "combined")
    combined.write_declaration_record(DeclarationRecord(first_ref))
    combined.write_declaration_record(DeclarationRecord(second_ref))
    with pytest.raises(Exception, match="incompatible ObjectId mappings"):
        combined.query().object_refs().collect()

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first.write_declaration_record(DeclarationRecord(first_ref))
    second.write_declaration_record(DeclarationRecord(second_ref))
    for stores in ((first, second), (second, first)):
        with pytest.raises(Exception, match="object-reference authority conflicts"):
            Repo(stores).query().object_refs().collect()


def test_embedded_object_references_do_not_become_independent_authority(tmp_path):
    """Structural references may conflict without becoming declaration authority."""

    object_id = ObjectId(("embedded",))
    first_ref = ObjectRef(
        Definition(SourceLeaf, "first").concretize(),
        {GraphPath(): object_id},
    )
    second_ref = ObjectRef(
        Definition(SourceLeaf, "second").concretize(),
        {GraphPath(): object_id},
    )
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first.write_definition_record(DefinitionRecord(
        Definition(SourcePair, first_ref).concretize()
    ))
    second.write_definition_record(DefinitionRecord(
        Definition(SourcePair, second_ref).concretize()
    ))

    assert Repo((first, second)).query().object_refs().count() == 2


def test_exact_embedded_state_and_owned_projection_preserve_full_knowledge(tmp_path):
    """Exact live selection keeps embedded projections without claiming storage."""

    writer = Repo(DirStore(tmp_path / "writer"))
    root = SourcePair(SourceLeaf("child", repo=writer), repo=writer)
    state = writer.save_object(root)
    child = state.at(GraphPath((Parameter("child"),)))
    embedded = DirStore(tmp_path / "embedded", query_index="memory")
    embedded.write_definition_record(DefinitionRecord(
        Definition(SourcePair, state).concretize(),
    ))
    reopened = Repo(DirStore.open_existing(embedded.base_dir, query_index="memory"))
    cached = Repo(DirStore(tmp_path / "cache-only", query_index="memory"))
    cached.cache_strong(root)
    for query in (embedded.query(), reopened.query(), cached.query(),
                  Repo((embedded, writer.stores[0])).query()):
        fixed = query.collect()
        for target in (state, child):
            expected = fixed.query().sel(target).collect()
            live = query.sel(target).collect()
            assert tuple(live) == tuple(expected) == (target,)
            assert live.sources(target) == expected.sources(target)
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
    conflicting.write_declaration_record(DeclarationRecord(ObjectRef(
        Definition(SourceLeaf, "other").concretize(), state.object.objects,
    )))
    query = Repo((direct, conflicting)).query().sel(state)
    assert query.exists()
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
    embedded.write_definition_record(DefinitionRecord(Definition(SourcePair, state).concretize()))
    empty = DirStore(tmp_path / "empty", query_index="memory")
    assert embedded.query().sel(state).exists()
    assert not empty.query().sel(state).exists()
    for store in (embedded, empty):
        monkeypatch.setattr(store, "iter_definition_records", lambda: pytest.fail("forbidden fallback scanned"))
        with pytest.raises(QueryWouldScanError, match="inventory scan"):
            store.query().sel(state).scan_policy("forbid").exists()


def test_exact_direct_state_collect_completes_embedded_source_evidence(tmp_path):
    """Direct authority proves membership, not the complete contributor set."""

    direct = DirStore(tmp_path / "direct", query_index="memory")
    writer = Repo(direct)
    state = writer.save_object(SourceLeaf("saved", repo=writer))
    embedded = DirStore(tmp_path / "embedded", query_index="memory")
    embedded.write_definition_record(DefinitionRecord(Definition(SourcePair, state).concretize()))
    query = Repo((direct, embedded)).query()
    expected = query.collect().query().sel(state).collect()
    result = query.sel(state).collect()
    assert result.sources(state) == expected.sources(state)
    assert len(result.sources(state)) == 2
    assert query.sel(state).in_source(embedded).one() == state


def test_selected_combinations_and_replacement_tokens_do_not_hide_conflicts(tmp_path, monkeypatch):
    """Proof reuse is domain/token-local, including unchanged demand recapture."""

    from dryml.core.query import source as source_module
    from dryml.core.query.model import QueryError

    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    third = DirStore(tmp_path / "third", query_index="memory")
    ref = ObjectRef(Definition(SourceLeaf, "first").concretize(), {GraphPath(): ObjectId()})
    conflict = ObjectRef(Definition(SourceLeaf, "conflict").concretize(), ref.objects)
    first.write_declaration_record(DeclarationRecord(ref))
    third.write_declaration_record(DeclarationRecord(conflict))
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
    second.write_declaration_record(DeclarationRecord(conflict))
    from dryml.core.query.source import _SourceRecapture

    capture.active = True
    with pytest.raises(_SourceRecapture) as changed:
        capture.capture_store(second, metadata_scopes=frozenset(("object", "lineage")))
    assert changed.value.invalidated
    with pytest.raises(QueryError, match="authority conflicts"):
        capture.capture_stores((first, second))
    assert len(passes) == 4


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
    store.write_definition_record(DefinitionRecord(Definition(SourcePair, embedded).concretize()))
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
    monkeypatch.setattr(store, "read_metadata", lambda *_a: (_ for _ in ()).throw(PermissionError("private")))
    with pytest.raises(_SourceRecapture) as changed:
        capture.capture_store(store, metadata_scopes=frozenset(("object", "lineage")))
    assert changed.value.invalidated
    new = capture.capture_store(store)
    assert new.capture_token is not old.capture_token
    assert isinstance(new.captured_metadata(state.object, "object"), _MetadataReadFailure)
    assert tuple(new.knowledge()) == tuple(old.knowledge())
    query = store.query().sel(state).where(field("object", "missing").missing())
    with pytest.raises(QueryError, match="metadata read failed"):
        query._collect_with_capture(capture)


def test_zip_inventory_preserves_uncommitted_transaction_domains(tmp_path):
    """Same archive handles retain independent extracted authority and tokens."""

    archive = tmp_path / "transactions.zip"
    seed = ZipStore(archive)
    base = Definition(SourceLeaf, "base").concretize()
    added = Definition(SourceLeaf, "added").concretize()
    seed.write_definition_record(DefinitionRecord(base))
    seed.commit()
    seed.close()
    first, second = ZipStore.open_existing(archive), ZipStore.open_existing(archive)
    try:
        first.write_definition_record(DefinitionRecord(added))
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
        first._validated_definition_root_inventory()
