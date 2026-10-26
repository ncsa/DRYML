"""Focused U4 coverage for private typed Query V3 relationship traversal."""

from __future__ import annotations

import pytest

from dryml.core import (
    Definition,
    Generator,
    Object,
    ObjectId,
    LocalState,
    Repo,
    Serializable,
    StateKey,
)
from dryml.core.cdef_graph import EdgeKind
from dryml.core.domains import UniformFromSet
from dryml.core.links import DefLink
from dryml.core.query.identity import IdentitySet
from dryml.core.query import field
from dryml.core.query.identity import Occurrence, OccurrenceSet
from dryml.core.query.query import IdentityQuery
from dryml.core.query.relationships import (
    EdgePolicy,
    RelationshipDepthExceeded,
    RelationshipKind,
    RelationshipPath,
)
from dryml.core.store.dir import DirStore
from dryml.core.template import Par
from dryml.core.utils.graph.path import GraphPath, Parameter


class V3TraversalLeaf(Serializable):
    """Minimal owned stateful child used by detached reference fixtures."""

    def __init__(self, name):
        self.name = name


class V3TraversalParent(Object):
    """Minimal parent retaining one direct materialized child definition."""

    def __init__(self, child):
        self.child = child


class V3TemplateLeaf(Object):
    """Stateless nested node selected through exact Generator support."""

    def __init__(self, width):
        self.width = width


class V3TemplateParent(Object):
    """Stateless root retaining generator-selected child topology."""

    def __init__(self, children):
        self.children = children


class V3DiamondNode(Object):
    """Labeled shared node for existential traversal work counters."""

    def __init__(self, label, children):
        self.label = label
        self.children = children


def _references():
    """Build a saved Node and its direct owned child projections."""

    child = Definition(V3TraversalLeaf, "child").concretize()
    root = Definition(V3TraversalParent, child).concretize()
    child_path = GraphPath((Parameter("child"),))
    object_ref = root.with_realization(identities={child_path: ObjectId(("root", "child"))})
    state = object_ref.with_realization(states={child_path: LocalState(
        StateKey(child.creation_digest(), "pkl-" + "1" * 64), child.creation_view())})
    return root, child, state, object_ref, state.at(child_path), object_ref.at(child_path), child_path


def test_u2_occurrences_projections_closure_share_and_fixed_results_detach(tmp_path, monkeypatch):
    from dryml.core.query.sqlite import SQLiteQueryIndexConfig

    root, child, *_ = _references()
    producer, scope = (DirStore(tmp_path / name, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
                       for name in ("producer", "scope"))
    producer.publish_node(root)
    calls = []
    for store in (producer, scope):
        store.rebuild_query_index()
        original = store.open_query_index().refresh

        def refresh(policy, *, stats=None, _original=original, _store=store, **kwargs):
            calls.append(_store)
            return _original(policy, stats=stats, **kwargs)

        monkeypatch.setattr(store.open_query_index(), "refresh", refresh)
    roots = producer.query().sel(root).refresh(True)
    nested = roots.nested(child)
    fixed_occurrences = nested.collect()
    assert calls == [producer]
    fixed_identities = roots.closure().collect()
    assert nested.owners().one().graph_equal(root) and nested.targets().one().graph_equal(child)
    assert calls == [producer]
    assert fixed_occurrences.query().count() == 1
    assert fixed_identities.query().count() == 2
    assert fixed_occurrences.query().where(field("object").missing(), scope=scope).count() == 0
    assert fixed_identities.query().stored(scope=scope).count() == 0
    assert calls == [producer]  # result evidence cannot carry live refresh intent
    scoped = nested.where(field("object").missing(), scope=scope)
    assert scoped.count() == 0
    assert calls == [producer, scope]
    assert scoped.owners().count() == scoped.targets().count() == 0
    assert calls == [producer, scope]
    assert scoped.targets().refresh(True).count() == 0
    assert calls == [producer, scope, producer, scope]


def test_v3_nested_requires_a_non_empty_typed_path_and_keeps_association_hops():
    root, _, state, object_ref, child_state, _, child_path = _references()
    query = IdentityQuery.from_set(IdentitySet((state,)))

    assert query.nested(state).count() == 0
    association = query.nested(object_ref, edges=EdgePolicy.STATE_OBJECT).one()
    assert association.path == RelationshipPath.from_hops(
        ((RelationshipKind.STATE_OBJECT, GraphPath()),),
    )
    occurrence = query.nested(child_state).one()
    assert occurrence.owner == state
    assert occurrence.target == child_state
    assert occurrence.path == RelationshipPath.from_hops(
        ((RelationshipKind.MATERIALIZE, child_path),),
    )
    assert occurrence.path.graph_path == child_path
    assert root not in {occurrence.target}
    with pytest.raises(RelationshipDepthExceeded, match="depth budget"):
        query.nested(child_state).max_depth(0).count()
    assert query.nested(object_ref).path(GraphPath()).count() == 1
    assert query.nested(object_ref).path(RelationshipPath()).count() == 0
    assert IdentityQuery.from_set(IdentitySet((object_ref,))).closure(
        EdgePolicy.STATE_OBJECT
    ).one() == object_ref


def test_v3_all_policy_keeps_reference_edges_and_shared_paths_distinct():
    child = Definition(V3TraversalLeaf, "child").concretize()
    reference_root = Definition(
        V3TraversalParent, DefLink.finalized(EdgeKind.REF, child),
    ).concretize()
    shared_root = Definition(V3TraversalParent, [child, child]).concretize()

    reference_query = IdentityQuery.from_set(IdentitySet((reference_root,))).nested(child)
    shared = IdentityQuery.from_set(IdentitySet((shared_root,))).nested(child).collect()

    assert reference_query.count() == 1
    assert reference_query.through(RelationshipKind.REFERENCE).count() == 1
    assert reference_query.max_occurrences(1).collect().bounded
    assert IdentityQuery.from_set(IdentitySet((reference_root,))).nested(
        child, edges=EdgePolicy.OWNED,
    ).count() == 0
    assert shared.count() == 2
    assert len({item.path for item in shared}) == 2


def test_v3_occurrence_kind_restrictions_only_narrow_targets():
    """Chained incompatible kind restrictions produce an empty intersection."""

    root, child, _, _, _, _, child_path = _references()
    occurrences = OccurrenceSet((Occurrence(root, child_path, child),))

    assert occurrences.query().cdefs().count() == 1
    assert occurrences.query().cdefs().object_refs().count() == 0
    assert occurrences.query().target_kind("cdef").state_refs().count() == 0


def test_stateless_owner_projection_preserves_each_source_domain(tmp_path):
    """Equal-content recipe/object/saved roots keep domain-specific provenance."""
    recipe, saved = (DirStore(tmp_path / name, query_index="sqlite") for name in ("recipe", "saved"))
    child = Definition(V3TemplateLeaf, 64).concretize()
    root = Definition(V3TemplateParent, child).concretize()
    recipe.publish_node(root)
    saved.publish_node(root, kind="saved")
    owners = Repo((recipe, saved)).query().sel(root).nested(child).owners().collect()
    assert owners.count() == 3
    assert owners.query().cdefs().count() == 1
    assert owners.query().object_refs().count() == 1
    assert owners.query().saved().count() == 1
    entries = {key.kind: evidence.sources for key, (_, evidence) in owners._entries.items()}
    assert entries["cdef"] == frozenset((IdentityQuery._source_evidence(recipe), IdentityQuery._source_evidence(saved)))
    assert entries["object_ref"] == entries["state_ref"] == frozenset((IdentityQuery._source_evidence(saved),))


def test_v3_direct_projections_ignore_raw_caps_without_enumerating_occurrences(monkeypatch):
    root, child, _, _, _, _, _ = _references()
    query = IdentityQuery.from_set(IdentitySet((root,))).nested(child).max_occurrences(0)

    monkeypatch.setattr(
        "dryml.core.query.relationships.iter_relationship_occurrences",
        lambda *_args, **_kwargs: pytest.fail("direct projection enumerated raw occurrences"),
    )

    assert query.owners().one() == root
    assert query.targets().one() == child


def test_v3_direct_projection_visits_each_diamond_state_once(monkeypatch):
    """Existential projection stays linear over exponentially many raw paths."""

    from dryml.core.query import relationships

    target = Definition(V3TraversalLeaf, "target").concretize()
    layer = (target, target)
    nodes = {id(target)}
    for depth in range(7):
        layer = tuple(
            Definition(
                V3DiamondNode,
                f"{depth}-{side}",
                layer,
            ).concretize()
            for side in ("left", "right")
        )
        nodes.update(map(id, layer))
    root = Definition(V3DiamondNode, "root", layer).concretize()
    nodes.add(id(root))
    query = IdentityQuery.from_set(IdentitySet((root,))).nested(target)
    visits = []
    original = relationships._ordered_direct_edges

    def counted(value):
        visits.append(id(value))
        return original(value)

    monkeypatch.setattr(relationships, "_ordered_direct_edges", counted)
    monkeypatch.setattr(
        relationships,
        "iter_relationship_occurrences",
        lambda *_args, **_kwargs: pytest.fail("direct projection built raw paths"),
    )

    assert query.targets().one() == target
    assert len(visits) <= len(nodes)
    assert len(visits) == len(set(visits))


def test_v3_bounded_roots_stay_bounded_through_refinement_and_relationships():
    root, child, state, _, _, _, _ = _references()
    roots = IdentityQuery.from_set(IdentitySet((state,), bounded=True))

    assert roots.state_refs().collect().bounded
    assert roots.closure().collect().bounded
    assert roots.nested(child).collect().bounded
    assert roots.nested(child).owners().collect().bounded
    assert roots.nested(child).targets().collect().bounded


def test_v3_occurrence_take_returns_fixed_canonical_bounded_prefix(monkeypatch):
    from dryml.core.query.query import OccurrenceQuery

    child = Definition(V3TraversalLeaf, "child").concretize()
    root = Definition(V3TraversalParent, [child, child]).concretize()
    query = IdentityQuery.from_set(IdentitySet((root,))).nested(child)
    expected = tuple(query.collect())[:1]
    monkeypatch.setattr(
        OccurrenceQuery, "collect",
        lambda self: pytest.fail("take() built a complete occurrence set"),
    )

    prefix = query.take(1)
    assert isinstance(prefix, OccurrenceSet)
    assert prefix.bounded and prefix.requested_limit == 1
    assert tuple(prefix) == expected
    assert query.take(0).count() == 0


def test_v3_occurrence_metadata_filters_targets_with_shared_authority(tmp_path):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = V3TraversalLeaf("child", repo=repo)
    state = repo.save_object(
        child, annotations=SaveAnnotations(object={"team": "selected"}),
    )
    roots = IdentityQuery.from_set(IdentitySet((state,)))
    nested = roots.nested().where(field("object", "team").eq("selected"), scope=store)

    assert nested.targets().one().graph_equal(state.object_view())
    assert nested.one().target.graph_equal(state.object_view())
    assert roots.nested().where(field("object", "team").eq("missing"), scope=store).count() == 0


def test_v3_fixed_occurrence_metadata_requires_explicit_scope(tmp_path):
    """Fixed paths stay detached while explicit scope metadata is fresh per terminal."""

    from dryml.core import Repo, SaveAnnotations
    from dryml.core.query.model import QueryDomainError
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        V3TraversalLeaf("child", repo=repo),
        annotations=SaveAnnotations(object={"team": "selected"}),
    )
    result = OccurrenceSet((Occurrence(state, GraphPath((Parameter("edge"),)), state.object_view()),))

    with pytest.raises(QueryDomainError, match="explicit scope"):
        result.query().where(field("object", "team").eq("selected"))
    query = result.query().where(field("object", "team").eq("selected"), scope=store)
    assert query.one().target.graph_equal(state.object_view())
    assert query.targets().one().graph_equal(state.object_view())
    repo.set_metadata(state.object_view(), {"team": "changed"}, store=store, scope="object")
    assert query.count() == 0
    assert query.targets().count() == 0
    assert list(query.iter()) == []
    assert result.count() == 1


def test_v3_occurrence_selector_cannot_hide_invalid_later_metadata(tmp_path):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    invalid = repo.save_object(
        V3TraversalLeaf("invalid", repo=repo),
        annotations=SaveAnnotations(object={"score": "wrong-type"}),
    )
    selected = repo.save_object(
        V3TraversalLeaf("selected", repo=repo),
        annotations=SaveAnnotations(object={"score": 1}),
    )
    owner = Definition(V3TraversalParent, None).concretize()
    fixed = OccurrenceSet((
        Occurrence(owner, GraphPath((Parameter("first"),)), invalid.object_view()),
        Occurrence(owner, GraphPath((Parameter("second"),)), selected.object_view()),
    ))

    with pytest.raises(Exception, match="numeric field"):
        fixed.query().sel(selected.object_view()).where(field("object", "score").lt(3), scope=store).exists()

    assert fixed.query().sel(selected.object_view()).sel(invalid.object_view()).count() == 0


def test_v3_occurrence_late_metadata_demand_restarts_root_evidence(tmp_path, monkeypatch):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        V3TraversalLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": "old"}),
    )
    lineage = repo.get_lineage_metadata(state.object_view())
    roots = IdentityQuery.from_store(store).state_refs().where(
        field("object", "team").eq("old")
    )
    original = IdentityQuery._collect_with_capture
    mutated = False

    def move_after_root(self, capture):
        nonlocal mutated
        result = original(self, capture)
        if self is roots and not mutated:
            mutated = True
            repo.set_metadata(state.object_view(), {"team": "new"}, store=store, scope="object")
        return result

    monkeypatch.setattr(IdentityQuery, "_collect_with_capture", move_after_root)
    nested = roots.nested().where(field("object", "team").eq("new")).where(
        field("lineage", "creation_status").eq(lineage.creation_status)
    )

    from dryml.core.query.model import QueryError
    with pytest.raises(QueryError, match="changed"):
        nested.count()
    assert mutated


def _v3_template_selector(*, shared: bool):
    child = Definition(V3TemplateLeaf, Par("width"))
    children = (
        [child, child]
        if shared else [child, Definition(V3TemplateLeaf, Par("width"))]
    )
    return Generator(
        Definition(V3TemplateParent, children),
        {"width": UniformFromSet((64,))},
    ).support_selector()


def _v3_template_leaf_selector():
    return Generator(
        Definition(V3TemplateLeaf, Par("width")),
        {"width": UniformFromSet((64,))},
    ).support_selector()


def test_v3_nested_generator_preserves_shared_topology_witnesses(tmp_path):
    """Nested exact support distinguishes shared and merely equal child graphs."""

    repo = Repo(DirStore(tmp_path / "store"))
    shared_leaf = V3TemplateLeaf(64, repo=repo)
    shared = V3TemplateParent([shared_leaf, shared_leaf], repo=repo)
    independent = V3TemplateParent(
        [V3TemplateLeaf(64, repo=repo), V3TemplateLeaf(64, repo=repo)],
        repo=repo,
    )
    root = V3TemplateParent(
        [shared.definition, independent.definition], repo=repo,
    )
    repo.save_object(root)

    result = repo.query().stored().cdefs().nested(
        _v3_template_selector(shared=True)
    ).targets().collect()

    assert len(result) == 1 and result.one().graph_equal(shared.definition)


def test_v3_nested_generator_honors_reference_aware_edge_policies(tmp_path):
    """Generator targets retain literal materializing and reference paths."""

    repo = Repo(DirStore(tmp_path / "store"))
    material = V3TemplateParent(V3TemplateLeaf(64, repo=repo), repo=repo)
    reference = V3TemplateParent(
        DefLink.finalized(
            EdgeKind.REF,
            V3TemplateLeaf(64, repo=repo).definition,
        ),
        repo=repo,
    )
    mixed = V3TemplateParent(
        DefLink.finalized(
            EdgeKind.REF,
            V3TemplateParent(V3TemplateLeaf(64, repo=repo), repo=repo).definition,
        ),
        repo=repo,
    )
    for value in (material, reference, mixed):
        repo.save_object(value)

    roots = repo.query().stored().cdefs()
    selector_value = _v3_template_leaf_selector()
    references = EdgePolicy(frozenset((RelationshipKind.REFERENCE,)))
    all_occurrences = roots.nested(selector_value)

    assert roots.nested(selector_value, edges=EdgePolicy.OWNED).count() == 1
    assert roots.nested(selector_value, edges=references).count() == 1
    assert all_occurrences.count() == 3
    assert all_occurrences.through(RelationshipKind.REFERENCE).count() == 2


@pytest.mark.parametrize("projection", ("owners", "targets"))
@pytest.mark.parametrize("sidecar", ("missing", "dirty"))
def test_u6_projection_disabled_refresh_survives_every_terminal(tmp_path, monkeypatch, projection, sidecar):
    """Projection inherits disabled refresh with no sidecar mutation."""

    from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex

    root, child, *_ = _references()
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(root)
    store.rebuild_query_index()
    index = store.open_query_index()
    if sidecar == "missing":
        index.close()
        index.path.unlink()
    else:
        index._mark_dirty()
    calls = []
    monkeypatch.setattr(SQLiteStoreQueryIndex, "rebuild", lambda *a, **kw: calls.append("rebuild"))
    monkeypatch.setattr(SQLiteStoreQueryIndex, "initialize_empty", lambda *a: calls.append("initialize"))
    query = getattr(store.query().cdefs().stored().refresh(False).nested(child), projection)()
    for terminal in (query.exists, query.count, query.one, query.one_or_none, query.collect,
                     lambda: query.take(1), lambda: query.explain(analyze=True)):
        from dryml.core.query.model import QueryIndexDirty, QueryIndexUnavailable
        with pytest.raises((QueryIndexDirty, QueryIndexUnavailable)):
            terminal()
        assert calls == []
    assert query.refresh_policy is False


def test_u6_projection_outer_budget_depth_and_witness_controls():
    """Targets, roots, and traversal share the effective outer terminal controls."""

    from dryml.core.errors import ParameterizationLimitError
    from dryml.core.query.model import QueryVerifyBudgetExceeded

    child = Definition(V3TemplateLeaf, 64).concretize()
    other = Definition(V3TemplateLeaf, 32).concretize()
    root = Definition(V3TemplateParent, [child, other]).concretize()
    roots = IdentitySet((root,)).query().max_verify(0).max_depth(0).max_witnesses(1)
    projected = roots.nested(_v3_template_leaf_selector()).targets()
    with pytest.raises(QueryVerifyBudgetExceeded):
        projected.collect()
    with pytest.raises(RelationshipDepthExceeded):
        projected.max_verify(3).collect()
    with pytest.raises(ParameterizationLimitError):
        projected.max_verify(3).max_depth(None).collect()
    query = projected.max_verify(3).max_depth(None).max_witnesses(None)
    assert query.one() == child
    assert query.explain(analyze=True).verified_count == 3
    assert roots.closure().max_verify(3).max_depth(None).count() == 3


def test_u6_occurrence_external_target_scope_cannot_escape_controls(tmp_path, monkeypatch):
    """Occurrence targets and both projections guard external metadata IO."""

    from dryml.core.query.model import QueryWouldScanError

    root, child, *_ = _references()
    store = DirStore(tmp_path / "scope", query_index="memory")
    scans = []
    monkeypatch.setattr("dryml.core.query.authority.capture_store_projection", lambda *args, **kwargs: scans.append(1))
    roots = IdentitySet((root,)).query().scan_policy("forbid")
    occurrence = roots.nested(child).where(field("object").missing(), scope=store)
    for terminal in (occurrence.exists, occurrence.owners().exists, occurrence.targets().exists):
        with pytest.raises(QueryWouldScanError):
            terminal()
        assert scans == []


@pytest.mark.parametrize("projection", ("direct", "owners", "targets"))
@pytest.mark.parametrize("domain", ("positive", "empty-targets", "empty-roots"))
def test_occurrence_combines_producer_and_external_metadata_authority(tmp_path, projection, domain):
    """Unrelated root conflicts precede root/target filtering in every terminal."""

    from dryml.core.query.model import QueryError

    producer = DirStore(tmp_path / "producer", query_index="memory")
    metadata = DirStore(tmp_path / "metadata", query_index="memory")
    target = Definition(V3TraversalLeaf, "target").concretize().with_identity(ObjectId())
    owner = Definition(V3TraversalParent, target).concretize()
    producer.publish_node(owner)
    metadata.publish_node(target)
    Repo(metadata).set_metadata(target, {"team": "selected"}, scope="object")
    object_id = ObjectId()
    for store, name in ((producer, "first"), (metadata, "second")):
        store.publish_node(Definition(V3TraversalLeaf, name).concretize().with_identity(object_id))
    roots = producer.query().cdefs().sel(
        Definition(V3TraversalParent, None).concretize() if domain == "empty-roots" else owner
    )
    query = roots.nested(target).where(
        field("object", "team").eq("absent" if domain == "empty-targets" else "selected"),
        scope=Repo(metadata) if domain == "empty-roots" else metadata,
    )
    if projection != "direct":
        query = getattr(query, projection)()
    for terminal in (query.exists, query.count, query.one, query.one_or_none, query.collect,
                     lambda: query.take(1)):
        with pytest.raises(QueryError, match="authority conflicts"):
            terminal()


def test_occurrence_metadata_preflight_does_not_admit_or_project_scope_holders(tmp_path):
    """External holder inventory proves authority without widening candidate work."""


    producer = DirStore(tmp_path / "producer", query_index="sqlite")
    metadata = DirStore(tmp_path / "metadata", query_index="memory")
    target = Definition(V3TemplateLeaf, 64).concretize().with_identity(ObjectId())
    owner = Definition(V3TraversalParent, DefLink.finalized(EdgeKind.REF, target)).concretize()
    producer.publish_node(owner)
    metadata.publish_node(target)
    Repo(metadata).set_metadata(target, {"team": "selected"}, scope="object")
    for name in ("unrelated-a", "unrelated-b"):
        metadata.publish_node(Definition(V3TraversalLeaf, name).concretize().with_identity(ObjectId()))
    query = producer.query().cdefs().stored().sel(owner).max_verify(3).nested(target).where(
        field("object", "team").eq("selected"), scope=metadata,
    )
    assert query.one().target.graph_equal(target)
    assert query.owners().one().graph_equal(owner)
    result = query.targets().collect()
    assert result.one().graph_equal(target)
    assert result.sources(result.one()) == frozenset((IdentityQuery._source_evidence(producer),))
    assert query.targets().explain(analyze=True).verified_count == 3
def test_node_pattern_alias_partition_roles_and_exact_quotation():
    """Independent equal Nodes stay independent across Mat/Ref and quote barriers."""
    from dryml.core import Definition, Mat, Ref, QueryPredicate, AnyValue, SKIP_ARGS

    child = Definition(value=2)
    shared = Definition(first=Mat(child), second=Ref(child))
    independent = Definition(first=Mat(Definition(value=2)), second=Ref(Definition(value=2)))
    assert QueryPredicate(shared).matches(shared.copy_graph())
    assert not QueryPredicate(shared).matches(independent)
    assert not QueryPredicate(independent).matches(shared)
    assert not QueryPredicate(shared).matches(Definition(first=Ref(child), second=Mat(child)))
    assert not QueryPredicate(Definition(recipe=child.quote())).matches(
        Definition(recipe=Definition(value=2, extra=3).quote()))
    a, b = Definition(value=2), Definition(value=2)
    unordered = Definition(children={a, b}, selected=a)
    assert QueryPredicate(unordered).matches(unordered.copy_graph())
    assert not QueryPredicate(unordered).matches(Definition(children={a, b}, selected=Definition(value=2)))
    assert QueryPredicate(Definition(value=AnyValue())).matches(Definition(value=Ref(a)))
    assert QueryPredicate(Definition(value=AnyValue())).matches(Definition(value=a.quote()))
    assert not QueryPredicate(Definition(value=Definition(SKIP_ARGS).quote())).matches(
        Definition(value=Definition().quote()))


def test_wide_ordered_node_pattern_preserves_matcher_order_and_failures():
    from dryml.core import QueryPredicate, Satisfies

    calls = []
    mode = "accept"

    def ordered(value):
        calls.append(value)
        if value == 129:
            if mode == "raise":
                raise ValueError("ordered matcher failure")
            return mode != "reject"
        return True

    matcher = QueryPredicate(Definition(values=[Satisfies(ordered) for _ in range(256)])).compile()
    target = Definition(values=list(range(256)))
    assert matcher.matches(target)
    assert calls == list(range(256))
    calls.clear()
    mode = "reject"
    assert not matcher.matches(target)
    assert calls == list(range(130))
    calls.clear()
    mode = "raise"
    with pytest.raises(ValueError, match="ordered matcher failure"):
        matcher.matches(target)
    assert calls == list(range(130))
