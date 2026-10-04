"""Focused U3 coverage for private composable Query V3 restrictions."""

from __future__ import annotations

import pytest

from dryml.core import Definition, Object, Repo, SaveAnnotations, Selector, Serializable
from dryml.core.query import field
from dryml.core.query.identity import IdentitySet
from dryml.core.query.model import QueryDomainError
from dryml.core.query.query import IdentityQuery
from dryml.core.store.dir import DirStore


class V3Leaf(Serializable):
    """Small stateful fixture that supplies an exact StateRef."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from pathlib import Path

        Path(dest_dir, "value").write_text(self.value, encoding="utf-8")


class V3Parent(Object):
    """Fixture with a retained owned child for proper-subtree restrictions."""

    def __init__(self, child, optional="default"):
        self.child = child
        self.optional = optional


def _saved(repo, value, *, object_metadata=None, state_metadata=None):
    return repo.save_object(
        V3Leaf(value, repo=repo),
        annotations=SaveAnnotations(object=object_metadata, state=state_metadata),
    )


def test_v3_structural_and_exact_selection_preserve_fixed_candidate_universe(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = _saved(repo, "first")
    second = _saved(repo, "second")

    structural = IdentityQuery.from_store(repo.stores[0]).sel(Definition(V3Leaf, "first"))
    exact = IdentityQuery.from_store(repo.stores[0]).sel(first.object.definition)

    assert set(structural.collect()) == {first.object.definition, first.object, first}
    assert set(exact.collect()) == {first.object.definition, first.object, first}
    fixed = IdentityQuery.from_set(IdentitySet((first.object,))).sel(Definition(V3Leaf, "second"))
    assert fixed.count() == 0
    assert second.object not in fixed.collect()


def test_v3_reference_filters_are_subsets_and_namespace_validation_allocates_no_identity(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = _saved(repo, "value")
    query = IdentityQuery.from_store(repo.stores[0])
    child = V3Leaf("child", repo=repo)
    parent = V3Parent(child, repo=repo)
    parent_state = repo.save_object(parent)

    assert query.object_id(state.object.object_id).object_refs().one() == state.object
    assert query.state_hash(next(iter(state.states.values()))).state_refs().one() == state
    assert query.contains(child.object_ref).object_refs().one() == parent_state.object
    with pytest.raises(ValueError, match="namespace"):
        query.namespace(("bad-part",))


def test_v3_exact_state_selection_uses_the_digest_addressed_authority_path(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = _saved(repo, "value")
    monkeypatch.setattr(
        store, "iter_state_ref_records", lambda: pytest.fail("exact StateRef performed an inventory scan")
    )

    assert IdentityQuery.from_store(store).sel(state).state_refs().one() == state


def test_v3_controls_report_or_reject_required_inventory_scans(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = _saved(repo, "value")

    explanation = IdentityQuery.from_store(store).explain()
    assert explanation.scan_required
    with pytest.raises(Exception, match="index coverage"):
        IdentityQuery.from_store(store).require_indexed().collect()
    with pytest.raises(Exception, match="inventory scan"):
        IdentityQuery.from_store(store).scan_policy("forbid").collect()
    exact = IdentityQuery.from_store(store).sel(state).state_refs().scan_policy("forbid")
    assert exact.explain().fast_path == "exact-state-ref"
    assert exact.one() == state


def test_v3_index_only_does_not_claim_direct_authority_reads_are_indexed(tmp_path):
    store = DirStore(tmp_path / "store")
    state = _saved(Repo(store), "value")

    with pytest.raises(Exception, match="index"):
        IdentityQuery.from_store(store).sel(state).require_indexed().one()


def test_v3_strict_selector_policy_is_not_dropped(tmp_path):
    class V3Child(V3Leaf):
        pass

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(V3Child("value", repo=repo))
    selector = Selector(Definition(V3Leaf), strict=True)

    assert IdentityQuery.from_store(store).sel(Definition(V3Leaf)).count() > 0
    assert IdentityQuery.from_store(store).sel(selector).count() == 0


def test_v3_metadata_uses_captured_facts_and_keeps_state_negation_ineligible(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = _saved(
        repo,
        "value",
        object_metadata={"empty": {}, "null": None, "enabled": True},
        state_metadata={"score": 7},
    )
    predicate = field("object", "empty").eq({}) & field("object", "null").eq(None)
    query = IdentityQuery.from_store(store).where(predicate)
    monkeypatch.setattr(repo, "get_metadata", lambda *_args, **_kwargs: pytest.fail("V3 used a live Repo getter"))

    assert set(query.collect()) == {state.object, state}
    state_only = IdentityQuery.from_store(store).where(~field("state", "score").missing())
    assert tuple(state_only.collect()) == (state,)


def test_v3_metadata_read_failure_is_not_a_missing_field(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = _saved(repo, "value")
    monkeypatch.setattr(
        store, "read_metadata",
        lambda target: (_ for _ in ()).throw(PermissionError("private-path")),
    )

    with pytest.raises(Exception, match="metadata read failed") as error:
        IdentityQuery.from_store(store).where(field("object", "team").missing()).collect()
    assert "private-path" not in str(error.value)
    assert error.value.__context__ is None


def test_v3_metadata_absence_captures_inventory_once(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    _saved(repo, "first")
    _saved(repo, "second")
    calls = []
    original = store.iter_definition_records

    def counted():
        calls.append(1)
        yield from original()

    monkeypatch.setattr(store, "iter_definition_records", counted)
    assert IdentityQuery.from_store(store).where(field("object", "team").missing()).exists()
    assert len(calls) == 1


def test_v3_earlier_structural_match_does_not_hide_later_metadata_error(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    _saved(repo, "first", object_metadata={"score": "invalid"})
    _saved(repo, "second", object_metadata={"score": 1})

    with pytest.raises(Exception):
        IdentityQuery.from_store(store).sel(Definition(V3Leaf, "second")).where(
            field("object", "score").lt(3)
        ).exists()


def test_v3_metadata_capture_reads_each_holder_fact_once_per_cut(tmp_path, monkeypatch):
    from collections import Counter

    from dryml.core.query.source import SourceCapture

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = V3Leaf("first", repo=repo)
    first = repo.save_object(value, annotations=SaveAnnotations(object={"team": "x"}))
    value.value = "second"
    second = repo.save_object(value)
    assert first != second
    calls = Counter()
    snapshot_calls = Counter()
    original = store.read_metadata
    original_snapshot = store.read_snapshot_metadata

    def counted(target):
        calls[(type(target), target.digest())] += 1
        return original(target)

    def counted_snapshot(digest):
        snapshot_calls[digest] += 1
        return original_snapshot(digest)

    monkeypatch.setattr(store, "read_metadata", counted)
    monkeypatch.setattr(store, "read_snapshot_metadata", counted_snapshot)
    facts = SourceCapture().capture_store(
        store, metadata_scopes=frozenset(("object", "state", "lineage", "snapshot")),
    )

    assert facts.captured_metadata(first.object, "object") is not None
    assert calls
    assert all(count == 1 for count in calls.values())
    assert snapshot_calls
    assert all(count == 1 for count in snapshot_calls.values())


def test_v3_metadata_lineage_and_snapshot_scopes_use_the_captured_store_cut(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: 0)
    state = _saved(repo, "value")
    lineage = repo.get_lineage_metadata(state.object)
    snapshot = repo.get_snapshot_metadata(state)

    assert IdentityQuery.from_store(store).where(
        field("lineage", "creation_status").eq(lineage.creation_status)
    ).object_refs().one() == state.object
    assert IdentityQuery.from_store(store).where(
        field("snapshot", "saved_at").eq(snapshot.saved_at)
    ).state_refs().one() == state


def test_v3_fixed_metadata_requires_explicit_scope_and_alias_is_live_per_terminal(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = _saved(repo, "first", object_metadata={"team": "vision"})
    second = _saved(repo, "second", object_metadata={"team": "platform"})
    fixed = IdentityQuery.from_set(IdentitySet((first.object, second.object)))

    with pytest.raises(Exception, match="explicit scope"):
        fixed.where(field("object", "team").eq("vision"))
    assert fixed.where(field("object", "team").eq("vision"), scope=store).one() == first.object
    repo.set_alias("chosen", first.object)
    query = IdentityQuery.from_store(store).alias("chosen").object_refs()
    assert query.one() == first.object
    repo.set_alias("chosen", second.object)
    assert query.one() == second.object


def test_v3_cached_restricts_input_members_with_explicit_repo_cache_scope(tmp_path):
    from dryml.core.query.identity import IdentitySet

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    strong = V3Leaf("strong", repo=repo)
    weak = V3Leaf("weak", repo=repo)
    repo.cache_strong(strong)
    repo.cache_weak(weak)
    candidates = IdentitySet((strong.definition, weak.definition))

    assert candidates.query().cached(scope=repo).count() == 2
    assert candidates.query().cached(scope=repo, weak=False).one() == strong.definition
    with pytest.raises(QueryDomainError, match="explicit Repo scope"):
        candidates.query().cached()
    assert IdentityQuery.from_store(store).cached(scope=repo).count() == 0


def test_v3_exact_repo_selection_retains_cache_only_state_receipt(tmp_path):
    source_repo = Repo(DirStore(tmp_path / "source"))
    value = V3Leaf("cached", repo=source_repo)
    state = source_repo.save_object(value)
    repo = Repo(DirStore(tmp_path / "empty"))
    repo.cache_strong(value)

    assert IdentityQuery.from_repo(repo).sel(state).state_refs().one() == state


def test_v3_source_scope_preserves_conflicts_without_reading_live_metadata(tmp_path):
    first_store = DirStore(tmp_path / "first")
    second_store = DirStore(tmp_path / "second")
    writer = Repo((first_store, second_store))
    value = V3Leaf("value", repo=writer)
    state = writer.save_object(
        value, store=first_store, annotations=SaveAnnotations(object={"team": "vision"})
    )
    writer.save_object(value, store=second_store, source_store=first_store)
    writer.set_metadata(state.object, {"team": "platform"}, store=second_store)
    query = IdentityQuery.from_repo(Repo((first_store, second_store))).where(
        field("object", "team").eq("vision")
    ).object_refs()

    with pytest.raises(Exception, match="conflicts"):
        query.collect()
    assert IdentityQuery.from_repo(Repo((first_store, second_store))).where(
        field("object", "team").eq("vision"), scope=first_store
    ).object_refs().one() == state.object


def test_v3_selector_value_edits_retain_exact_authority_without_widening(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = _saved(repo, "first")
    _saved(repo, "second")
    selector = Selector(first.object.definition)

    assert selector.categorical(drop_args=True).restore().root == first.object.definition
    assert (
        IdentityQuery.from_store(repo.stores[0])
        .sel(selector)
        .categorical(drop_args=True)
        .object_refs()
        .count()
        == 1
    )
    with pytest.raises(Exception, match="ConcreteDefinition"):
        Selector(Definition(V3Leaf, "first")).exact()
