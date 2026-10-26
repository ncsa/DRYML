"""Characterization tests for the retained non-query Repo lookup surface.

Query V3 cutover must not change item lookup, ``get()``, or explicit load
semantics while public query producers migrate separately.
"""

import dryml.core as core
import dryml.core.query as query
import pytest

from dryml.core import Definition, Object, ObjectId, Repo, Serializable, SKIP_ARGS
from dryml.core.store.dir import DirStore


class LookupLeaf(Object):
    """Small stored and cached object used to characterize Repo lookups."""

    def __init__(self, name):
        self.name = name


class LookupRoot(Object):
    """Root that retains a child definition without publishing it as a root."""

    def __init__(self, child):
        self.child = child


class LookupStatefulLeaf(Serializable):
    """Root ObjectRef fixture for alias-only non-query authority checks."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class LookupCountingStore(DirStore):
    """Store spy proving retained exact lookups do not enumerate all definitions."""

    def __init__(self, store):
        super().__init__(store.base_dir, query_index=store.query_index)
        self.definition_reads = 0

    def _read_file(self, path, record_type):
        from dryml.core.store.records import NodeRootRecord
        if record_type is NodeRootRecord:
            self.definition_reads = getattr(self, "definition_reads", 0) + 1
        return super()._read_file(path, record_type)

def test_retained_item_get_and_load_lookups_keep_their_candidate_universes(tmp_path):
    """Item/get retain known roots and cache entries; explicit loads stay structural."""

    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    child = LookupLeaf("child", repo=writer)
    root = LookupRoot(child, repo=writer)
    writer.save_object(root)

    reopened = Repo(DirStore(store.base_dir))

    assert reopened.query().sel(child.definition).cdefs().stored().count() == 0
    assert reopened.get(child.definition).count() == 0
    with pytest.raises(KeyError):
        reopened[child.definition]
    retained_root = reopened[root.definition]
    assert retained_root.definition.graph_equal(root.definition)
    assert reopened.get(root.definition).one().definition.graph_equal(root.definition)
    assert reopened.get(child.definition).count() == reopened.query().sel(child.definition).cdefs().count()
    with pytest.raises(TypeError, match="ConcreteDefinition"):
        reopened[Definition(LookupRoot, SKIP_ARGS)]

    # The child definition record is structural authority but not a get/item
    # candidate until it is an explicitly retained cache member.
    assert reopened.load_object(child.definition).definition.graph_equal(child.definition)
    missing = LookupLeaf("missing", repo=reopened).definition
    assert reopened.load_object(missing).definition.graph_equal(missing)

    reopened.cache_strong(child)
    assert reopened.get(child.definition).one().definition.graph_equal(child.definition)
    assert reopened.get(lambda value: value is child).one() is child


def test_retained_exact_item_and_get_use_v3_direct_root_lookup(tmp_path, monkeypatch):
    """Exact retained lookups retain the old no-broad-hydration performance path."""

    store = DirStore(tmp_path / "store", query_index="memory")
    writer = Repo(store)
    root = LookupLeaf("root", repo=writer)
    writer.save_object(root)

    counting = LookupCountingStore(DirStore(store.base_dir, query_index="memory"))
    reopened = Repo(counting)
    monkeypatch.setattr("dryml.core.query.authority.capture_store_projection",
                        lambda *a, **k: pytest.fail("exact retained lookup inventoried Store metadata"))

    assert reopened[root.definition].definition.graph_equal(root.definition)
    assert reopened.get(root.definition).one().definition.graph_equal(root.definition)
    assert counting.definition_reads >= 1


def test_explicit_weak_cached_child_is_a_retained_get_candidate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = LookupLeaf("child", repo=repo)
    root = LookupRoot(child, repo=repo)
    repo.cache_weak(root)
    repo.cache_weak(child)

    assert repo.query().sel(child.definition).cdefs().cached().count() == 1
    assert repo.get(child.definition).one().definition.graph_equal(child.definition)
    assert repo[child.definition].definition.graph_equal(child.definition)
    assert repo.get(lambda value: value is child).count() == 0
    repo.cache_strong(child)
    assert repo.get(lambda value: value is child).one() is child


def test_retained_get_preserves_alias_only_root_candidate_policy(tmp_path):
    from dryml.core.store.records import NodeNameRecord

    store = DirStore(tmp_path / "store")
    cdef = Definition(LookupLeaf, "alias-only").concretize()
    record = store.publish_node(cdef, kind="saved")
    store.compare_and_set_node_name(NodeNameRecord("selected", "saved", record.digest), expected=None)
    repo = Repo(store)

    expected = repo.query().sel(cdef).cdefs().stored().count()
    assert expected == 0
    assert repo.get(cdef).count() == expected


def test_v3_producers_and_exports_replace_the_retired_query_surface(tmp_path):
    """Repo and Store produce V3 plans and retired public names are absent."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    root = LookupLeaf("root", repo=repo)
    repo.save_object(root)

    constructions = repo._num_constructions
    repo_query = repo.query()
    store_query = store.query()

    assert type(repo_query).__name__ == "IdentityQuery"
    assert type(store_query).__name__ == "IdentityQuery"
    assert repo_query.sel(root.definition).cdefs().stored().one().graph_equal(root.definition)
    assert store_query.sel(root.definition).cdefs().stored().one().graph_equal(root.definition)
    assert repo_query.take(1).bounded
    assert repo._num_constructions == constructions
    assert not hasattr(repo, "references")
    assert not hasattr(repo, "find_defs")
    assert not hasattr(core, "DefinitionQuery")
    assert not hasattr(core, "ReferenceQuery")
    assert not hasattr(query, "DefinitionQuery")
    assert not hasattr(query, "ReferenceQuery")
    assert core.ObjectResultSet is query.ObjectResultSet
    assert core.QueryDiagnostic is query.QueryDiagnostic
    assert not hasattr(repo_query.take(1), "refine")
    assert not hasattr(core, "ContainmentEdgePolicy")
    assert not hasattr(query, "ContainmentEdgePolicy")


def test_node_pattern_correlates_facets_on_the_same_occurrence():
    """A matching identity elsewhere cannot satisfy a state-constrained child."""
    from dryml.core import LocalState, StateKey
    from dryml.core.query.query import _query_match

    creation = Definition(LookupStatefulLeaf, "child").concretize()
    state = LocalState(StateKey(creation.creation_digest(), "pkl-" + "a" * 64), creation)
    identity = ObjectId(("selected",))
    child = Definition(LookupStatefulLeaf, name="child").with_identity(identity).with_state(state)
    pattern = Definition(LookupRoot, child=child)
    positive = Definition(LookupRoot, child=child.copy_graph(), unrelated=Definition().with_identity(ObjectId(("other",))))
    negative = Definition(LookupRoot, child=child.without_identity(), unrelated=child.without_state().copy_graph())
    match = lambda target: _query_match(pattern, target, strict=False, class_match="exact")
    assert match(positive)
    assert not match(negative)
    assert not match(Definition(LookupRoot, child=child.without_state()))


def test_explicit_predicate_absence_boolean_and_local_state_domain():
    """Absence is independent of omitted facets, wildcard presence and negation."""
    from dryml.core import AnyValue, Missing, LocalState, QueryPredicate, StateKey
    from dryml.core.query.model import QueryDomainError

    creation = Definition(LookupStatefulLeaf, "child").concretize()
    state = LocalState(StateKey(creation.creation_digest(), "pkl-" + "a" * 64), creation)
    bare = Definition(LookupStatefulLeaf, name="child")
    selected = bare.with_state(state).with_identity(ObjectId(("donor",)))
    missing = QueryPredicate.absent("state")
    assert missing.matches(bare)
    assert not missing.matches(selected)
    assert QueryPredicate(bare).matches(selected)
    assert (~missing & bare).matches(selected)
    assert (missing | selected).matches(bare)
    assert QueryPredicate(Definition(name=AnyValue())).matches(bare)
    assert not QueryPredicate(Definition(name=AnyValue())).matches(Definition())
    assert QueryPredicate(Definition()).matches(Definition())
    assert QueryPredicate(Definition(name=Missing())).matches(Definition())
    assert not QueryPredicate(Definition(name=Missing())).matches(bare)
    assert QueryPredicate(bare.with_state(state)).compile(domain="local_state").matches(state)
    assert QueryPredicate.absent("state").compile(domain="local_state").matches(state) is False
    other_bucket = LocalState(StateKey("b" * 64, state.key.codec_hash), creation)
    assert not QueryPredicate(bare.with_state(state)).compile(domain="local_state").matches(other_bucket)
    with pytest.raises(QueryDomainError, match="occurrence"):
        (missing | selected).compile(domain="local_state")
    assert QueryPredicate(selected).compile(domain="occurrence").matches(selected)


def test_node_pattern_compiler_retains_visible_exact_residual_without_effects(monkeypatch):
    """Legacy postings cannot certify negatives for unsupported Node facets."""
    from dryml.core import QueryPredicate
    from dryml.core.node import Node
    from dryml.core.query.graph_plan import graph_candidate_ids
    from dryml.core.query.model import QueryStats
    from dryml.core.query.selector_graph import compile_selector_graph

    pattern = Definition(LookupStatefulLeaf, name="child").with_identity(ObjectId(("selected",)))
    counts = dict.fromkeys(("construction", "allocation", "capture", "io", "defaults"), 0)
    def forbidden(kind):
        def fail(*args, **kwargs):
            counts[kind] += 1
            pytest.fail(kind)
        return fail
    monkeypatch.setattr(Node, "concretize", forbidden("defaults"))
    monkeypatch.setattr(ObjectId, "__init__", forbidden("allocation"))
    monkeypatch.setattr(LookupStatefulLeaf, "__init__", forbidden("construction"))
    monkeypatch.setattr(Repo, "save_object", forbidden("capture"))
    monkeypatch.setattr(DirStore, "read_node", forbidden("io"))
    monkeypatch.setattr(DirStore, "read_state", forbidden("io"))
    compiled = compile_selector_graph(pattern)
    assert compiled.requires_scan and compiled.residual_reasons
    assert compiled.exact_residual.matches(pattern.copy_graph())
    assert not compiled.exact_residual.matches(pattern.without_identity())
    class Catalog:
        def all_definition_ids(self):
            return {1, 2}
    stats = QueryStats()
    assert graph_candidate_ids(Catalog(), compiled, None, stats=stats) == {1, 2}
    assert stats.scan_required and stats.scan_reason
    assert QueryPredicate(pattern).matches(pattern.copy_graph())
    assert counts == dict.fromkeys(counts, 0)


def test_node_predicates_refine_fixed_metadata_without_source_capture_or_recompilation(monkeypatch):
    """Fixed recipe metadata can qualify the semantics before format-4 source cutover."""
    import dryml
    from dryml.core import IdentitySet, QueryPredicate, IntRange
    from dryml.core.query.node_pattern import _PreparedPredicate
    first = Definition(LookupLeaf, "first").concretize()
    second = Definition(LookupLeaf, "second").concretize()
    fixed = IdentitySet((first, second))
    predicate = QueryPredicate(Definition(name="first")) | Definition(name=IntRange(1, 2))
    selected = fixed.query().sel(predicate)
    assert dryml.QueryPredicate is QueryPredicate is query.QueryPredicate
    monkeypatch.setattr(_PreparedPredicate, "__init__", lambda *a, **k: pytest.fail("pattern recompiled per candidate"))
    assert selected.one().graph_equal(first)
