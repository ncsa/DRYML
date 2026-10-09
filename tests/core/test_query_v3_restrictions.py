"""Focused U3 coverage for private composable Query V3 restrictions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dryml.core import (
    Definition,
    Generator,
    Object,
    Repo,
    SaveAnnotations,
    Selector,
    Serializable,
    StateRef,
    Template,
    selector,
)
from dryml.core.domains import UniformFromSet
from dryml.core.errors import ParameterizationLimitError
from dryml.core.factory import FactorySpec
from dryml.core.params import AnyValue
from dryml.core.template import Par
from dryml.core.query import Arg, GraphPath, Kwarg, QueryError, QueryIndexUnavailable, field
from dryml.core.query.identity import IdentitySet, SourceEvidence
from dryml.core.query.model import QueryDomainError
from dryml.core.query.query import IdentityQuery
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore


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


class V3StateConsumer(Object):
    """Definition-only fixture that retains a selected StateRef."""

    def __init__(self, selected):
        self.selected = selected


class V3ConstructionTrap(Object):
    """Object type whose constructor must not run during V3 selection."""

    def __init__(self, value):
        raise AssertionError("V3 query selection must not construct Objects")


class V3FactoryOwner(Object):
    """Definition fixture retaining an inert partial FactorySpec pattern."""

    def __init__(self, factory):
        self.factory = factory


class V3TemplateOwner(Serializable):
    """Stateful fixture retaining one inert Template-role Definition."""

    def __init__(self, recipe: Template):
        self.recipe = recipe

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write one deterministic marker for exact-reference query coverage."""

        from pathlib import Path

        Path(dest_dir, "value").write_text("template-owner", encoding="utf-8")


class V3Variadic(Object):
    """Selector fixture with semantic var-positional and var-keyword paths."""

    def __init__(self, *values, **labels):
        self.values = values
        self.labels = labels


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


def test_resolved_definition_matches_equivalent_quoted_template_graph(tmp_path):
    """Authored Template values match their canonical quoted reference form."""

    repo = Repo(DirStore(tmp_path / "store"))
    authored = Definition(
        V3TemplateOwner,
        Definition(V3Leaf, "quoted"),
    )
    owner = repo.load_or_build(authored)
    state = repo.save_object(owner)
    exact = authored.concretize(repo=repo)
    exact_members = repo.query().sel(exact).collect()

    assert exact.graph_equal(state.definition)
    assert len(exact_members) == 3
    assert exact_members.query().sel(authored).count() == 3
    assert repo.query().sel(authored).count() == 3


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


def test_v3_object_ref_selector_keeps_its_existing_state_members(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = _saved(repo, "selected")
    other = _saved(repo, "other")
    older = StateRef(state.object, {path: "pkl-" + "0" * 64 for path in state.states})
    fixed = IdentitySet((state.object, state, older, other.object, other))

    assert fixed.query().sel(state.object).count() == 3
    assert fixed.query().sel(state.object).state_refs().count() == 2
    assert fixed.query().sel(state).one() == state


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
    assert exact.explain(analyze=True).source_cuts == 1
    assert exact.one() == state


def test_v3_refresh_and_scan_warning_controls_are_applied_at_terminals(
        tmp_path, monkeypatch):
    """Refresh executes once per terminal and warn reports a required scan."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    _saved(repo, "value")
    calls = []
    monkeypatch.setattr(
        repo._query_index,
        "refresh",
        lambda policy, **_kwargs: calls.append(policy),
    )

    with pytest.warns(RuntimeWarning, match="authoritative inventory scan"):
        assert repo.query().scan_policy("warn").exists()
    assert calls == ["auto"]
    assert repo.query().refresh(True).exists()
    assert calls == ["auto", True]
    assert repo.query().refresh(False).exists()
    assert calls == ["auto", True]
    with pytest.raises(QueryError, match="inventory scan"):
        repo.query().scan_policy("forbid").exists()
    assert calls == ["auto", True]


@pytest.mark.parametrize("value", (0, 1, None, [], "invalid"))
def test_v3_refresh_rejects_values_outside_its_exact_policy_domain(value):
    with pytest.raises(ValueError, match="refresh policy"):
        IdentitySet().query().refresh(value)


@pytest.mark.parametrize("value", (0, 1, None, [], "invalid"))
def test_v3_scan_policy_rejects_values_outside_its_string_domain(value):
    with pytest.raises(ValueError, match="scan policy"):
        IdentitySet().query().scan_policy(value)


def test_v3_repo_refresh_reports_only_completed_reconciliation(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    _saved(repo, "value")
    monkeypatch.setattr(repo._query_index, "refresh", lambda *_args, **_kwargs: False)

    assert repo.query().explain(analyze=True).refresh_action == "none"

    def unavailable(*_args, **_kwargs):
        raise QueryIndexUnavailable("synthetic unavailable index")

    monkeypatch.setattr(repo._query_index, "refresh", unavailable)
    with pytest.raises(QueryIndexUnavailable, match="synthetic unavailable index"):
        repo.query().refresh(True).exists()


def test_v3_repo_auto_refresh_tolerates_index_open_unavailability(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)

    def unavailable(_binding):
        raise QueryIndexUnavailable("synthetic unavailable index")

    monkeypatch.setattr(repo._query_index, "open_store_index", unavailable)
    assert repo._query_index.refresh("auto") is False
    with pytest.raises(QueryIndexUnavailable, match="synthetic unavailable index"):
        repo._query_index.refresh(True)


def test_v3_store_auto_refresh_tolerates_index_open_unavailability(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")

    def unavailable():
        raise QueryIndexUnavailable("synthetic unavailable index")

    monkeypatch.setattr(store, "open_query_index", unavailable)
    assert not store.query().exists()
    with pytest.raises(QueryIndexUnavailable, match="synthetic unavailable index"):
        store.query().refresh(True).exists()


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


def test_v3_metadata_keeps_scalar_types_and_validates_every_populated_leaf(tmp_path):
    """Typed equality and invalid ordering/containment cannot be short-circuited."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = _saved(
        repo,
        "typed",
        object_metadata={
            "bool": True,
            "int": 1,
            "float": 1.0,
            "text": "vision baseline",
            "tags": ["vision", "baseline"],
            "mapping": {"value": 1},
        },
    )
    selected = repo.query().sel(state.object).object_refs()

    assert selected.where(field("object", "bool").eq(1)).count() == 0
    assert selected.where(field("object", "int").eq(1.0)).count() == 0
    assert selected.where(
        field("object", "tags").eq(("vision", "baseline"))
    ).count() == 0
    assert selected.where(field("object", "text").contains("vision")).one() == state.object
    assert selected.where(field("object", "tags").contains("vision")).one() == state.object
    with pytest.raises(QueryError, match="numeric field"):
        selected.where(
            field("object", "text").lt(2)
            | field("object", "missing").eq("absent")
        ).collect()
    with pytest.raises(QueryError, match="containment"):
        selected.where(
            field("object", "mapping").contains("value")
            & field("object", "missing").eq("absent")
        ).collect()


def test_v3_unknown_and_epoch_lineage_times_remain_distinct(tmp_path, monkeypatch):
    """Unknown lineage does not compare as the Unix epoch under V3 filtering."""

    monkeypatch.setattr(
        "dryml.core.snapshot_capture.current_utc_time",
        lambda: datetime(1970, 1, 1, tzinfo=timezone.utc),
    )
    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: 0)
    unknown = repo.save_object(V3Parent(V3Leaf("child", repo=repo), repo=repo))
    known = _saved(repo, "known")

    assert repo.get_lineage_metadata(unknown.object).creation_status == "unknown"
    assert repo.query().sel(unknown.object).where(
        field("lineage", "created_at").ge(0)
    ).object_refs().count() == 0
    assert repo.query().sel(known.object).where(
        field("lineage", "created_at").eq(0)
    ).object_refs().one() == known.object


def test_v3_metadata_authority_has_zipstore_parity(tmp_path):
    """Buffered archive authority reopens with the same metadata query answer."""

    archive = tmp_path / "metadata.zip"
    store = ZipStore(archive)
    repo = Repo(store)
    state = _saved(repo, "zip", object_metadata={"team": "vision"})
    repo.close(flush=True)
    reopened = Repo(ZipStore.open_existing(archive))

    assert reopened.query().where(
        field("object", "team").eq("vision")
    ).object_refs().one() == state.object


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
    assert (
        IdentityQuery.from_store(repo.stores[0])
        .sel(first.object.definition)
        .categorical(drop_args=True)
        .exact()
        .cdefs()
        .stored()
        .one()
        == first.object.definition
    )
    with pytest.raises(Exception, match="ConcreteDefinition"):
        Selector(Definition(V3Leaf, "first")).exact()


def test_v3_selector_exact_translates_authored_paths_to_semantic_authority():
    """Exact edits map authored keyword paths onto ConcreteDefinition parameters."""

    root = V3Parent(V3Leaf("child")).definition
    edited = Selector(root).categorical(
        recursive=True, drop=("value",),
    ).exact(path="child")

    assert edited.root.parameters["child"] == root.parameters["child"]


def test_v3_selector_exact_retains_complete_variadic_semantic_paths():
    """Variadic authored paths keep Parameter plus Index or Key authority hops."""

    positional = V3Leaf("positional").definition
    keyword = V3Leaf("keyword").definition
    authored = Definition(V3Variadic, positional, named=keyword)
    exact = authored.concretize()
    selected = Selector(authored, exact_root=exact)

    assert selected.exact(path=GraphPath((Arg(0),))).root.args[0] == positional
    assert selected.exact(path=GraphPath((Kwarg("named"),))).root.kwargs["named"] == keyword


def test_v3_soft_state_selector_requires_explicit_scoped_preparation(tmp_path, monkeypatch):
    """Prepared soft aliases pin their named Repo authority before V3 composition."""

    repo = Repo(DirStore(tmp_path / "store"))
    value = V3Leaf("first", repo=repo)
    first = repo.save_object(value)
    value.value = "second"
    second = repo.save_object(value)
    repo.set_state_alias("chosen", first)
    source = Definition(V3StateConsumer, first.object.state("chosen"))

    with pytest.raises(TypeError, match=r"selector\(value, scope"):
        IdentityQuery.from_set(IdentitySet(())).sel(source)

    prepared = selector(source, scope=repo)
    assert prepared.root.parameters["selected"] == first
    repo.set_state_alias("chosen", second)
    monkeypatch.setattr(
        repo,
        "resolve_state_selector",
        lambda value: pytest.fail("prepared selector performed a hidden alias lookup"),
    )
    candidate = Definition(V3StateConsumer, first).concretize()

    assert IdentityQuery.from_set(IdentitySet((candidate,))).sel(prepared).one() == candidate


def test_v3_scoped_selector_failure_does_not_disclose_alias_or_upstream_error(tmp_path, monkeypatch):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(V3Leaf("candidate", repo=repo))
    secret = "query-v3-private-alias-sentinel"
    source = Definition(V3StateConsumer, state.object.state(secret))
    monkeypatch.setattr(
        repo, "resolve_state_selector",
        lambda value: (_ for _ in ()).throw(KeyError(secret)),
    )

    with pytest.raises(KeyError) as error:
        selector(source, scope=repo)
    assert secret not in str(error.value)
    assert error.value.__context__ is None


def test_v3_soft_alias_validation_does_not_skip_invalid_sibling(tmp_path):
    from dryml.core.selector import _contains_state_selector

    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(V3Leaf("candidate", repo=repo))
    cyclic = {}
    cyclic["self"] = cyclic

    with pytest.raises(ValueError, match="Cycle"):
        _contains_state_selector({"soft": state.object.state("missing"), "later": cyclic})


def test_v3_selection_rejects_live_objects_and_never_constructs_them():
    """V3 selection consumes retained identities and definitions, never live Objects."""

    candidate = Definition(V3ConstructionTrap, "value").concretize()

    assert (
        IdentityQuery.from_set(IdentitySet((candidate,)))
        .sel(Definition(V3ConstructionTrap, "value"))
        .one()
        == candidate
    )
    with pytest.raises(TypeError, match="Definition, selector, generator, or exact reference"):
        IdentityQuery.from_set(IdentitySet(())).sel(V3Leaf("value"))


def test_v3_generator_selection_keeps_exact_support_and_shared_witness_budget():
    """Generator support remains exact and exhausts one terminal witness budget."""

    first = Definition(V3Leaf, "first").concretize()
    second = Definition(V3Leaf, "second").concretize()
    selector_value = Generator(
        Definition(V3Leaf, Par("value")),
        {"value": UniformFromSet(("first",))},
    ).support_selector()
    query = IdentityQuery.from_set(IdentitySet((first, second))).sel(selector_value)

    assert query.one() == first
    with pytest.raises(ParameterizationLimitError, match="witness limit"):
        query.max_witnesses(1).collect()


def _shared_v3_parent_selector():
    child = Definition(V3Leaf, Par("value"))
    return Generator(
        Definition(V3Parent, [child, child]),
        {"value": UniformFromSet(("shared",))},
    ).support_selector()


def test_v3_generator_retains_matching_graph_witness_before_deduplication():
    """Generator selection distinguishes equal CDefs with different sharing."""

    shared_leaf = V3Leaf("shared")
    shared = V3Parent([shared_leaf, shared_leaf]).definition
    independent = V3Parent([V3Leaf("shared"), V3Leaf("shared")]).definition
    assert shared == independent
    assert not shared.graph_equal(independent)

    result = IdentityQuery.from_set(
        IdentitySet((independent, shared)),
    ).sel(_shared_v3_parent_selector()).collect()

    assert tuple(result) == (shared,)


@pytest.mark.parametrize("shared_first", [False, True])
def test_v3_generator_preserves_graph_witnesses_across_stores(
        tmp_path, shared_first):
    """Store ordering cannot select an equal but topologically wrong witness."""

    independent_store = DirStore(tmp_path / "independent")
    shared_store = DirStore(tmp_path / "shared")
    independent_repo = Repo(independent_store)
    independent = V3Parent(
        [
            V3Leaf("shared", repo=independent_repo),
            V3Leaf("shared", repo=independent_repo),
        ],
        repo=independent_repo,
    )
    independent_repo.save_object(independent)
    shared_repo = Repo(shared_store)
    leaf = V3Leaf("shared", repo=shared_repo)
    shared = V3Parent([leaf, leaf], repo=shared_repo)
    shared_repo.save_object(shared)
    stores = (
        (shared_store, independent_store)
        if shared_first else (independent_store, shared_store)
    )

    result = Repo(stores).query().stored().cdefs().sel(
        _shared_v3_parent_selector()
    ).collect()

    assert tuple(result) == (shared.definition,)
    assert result.sources(shared.definition) == frozenset((
        SourceEvidence.from_source(shared_store.authority_fence_key()),
    ))


def test_v3_partial_factory_pattern_uses_residual_scan_verification():
    """Wildcard FactorySpec arguments retain every matching fixed candidate."""

    first = V3FactoryOwner(FactorySpec("factory", 1)).definition
    second = V3FactoryOwner(FactorySpec("factory", 2)).definition
    selector_value = Definition(
        V3FactoryOwner,
        FactorySpec("factory", AnyValue()),
    )
    from dryml.core.query.selector_graph import compile_selector_graph

    graph = compile_selector_graph(selector_value)
    result = IdentityQuery.from_set(
        IdentitySet((first, second)),
    ).sel(selector_value).collect()

    assert graph is not None and graph.requires_scan
    assert set(result) == {first, second}
