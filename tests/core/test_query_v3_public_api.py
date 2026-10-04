"""Characterization tests for the retained non-query Repo lookup surface.

Query V3 cutover must not change item lookup, ``get()``, or explicit load
semantics while public query producers migrate separately.
"""

import pytest

from dryml.core import Definition, Object, ObjectId, ObjectRef, Repo, Serializable, SKIP_ARGS
from dryml.core.repo import RepoLoadError
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
        self.definition_enumerations = 0

    def read_definition_record(self, digest):
        self.definition_reads += 1
        return super().read_definition_record(digest)

    def iter_definition_records(self):
        self.definition_enumerations += 1
        return super().iter_definition_records()


def test_retained_item_get_and_load_lookups_keep_their_candidate_universes(tmp_path):
    """Item/get retain known roots and cache entries; explicit loads stay structural."""

    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    child = LookupLeaf("child", repo=writer)
    root = LookupRoot(child, repo=writer)
    writer.save_object(root)

    reopened = Repo(DirStore(store.base_dir))

    assert reopened.query(child.definition).known().defs().count() == 0
    assert reopened.get(child.definition).count() == 0
    with pytest.raises(KeyError):
        reopened[child.definition]
    retained_root = reopened[root.definition]
    assert retained_root.definition == root.definition
    assert reopened.get(root.definition).one().definition == root.definition
    assert reopened.get(child.definition).count() == reopened.query(child.definition).known().defs().count()
    with pytest.raises(TypeError, match="ConcreteDefinition"):
        reopened[Definition(LookupRoot, SKIP_ARGS)]

    # The child definition record is structural authority but not a get/item
    # candidate until it is an explicitly retained cache member.
    assert reopened.load_object(child.definition).definition == child.definition
    missing = LookupLeaf("missing", repo=reopened).definition
    with pytest.raises(RepoLoadError, match="No connected Store"):
        reopened.load_object(missing)

    reopened.cache_strong(child)
    assert reopened.get(child.definition).one().definition == child.definition
    assert reopened.get(lambda value: value is child).one() is child


def test_retained_exact_item_and_get_use_v3_direct_root_lookup(tmp_path):
    """Exact retained lookups retain the old no-broad-hydration performance path."""

    store = DirStore(tmp_path / "store", query_index="memory")
    writer = Repo(store)
    root = LookupLeaf("root", repo=writer)
    writer.save_object(root)

    counting = LookupCountingStore(DirStore(store.base_dir, query_index="memory"))
    reopened = Repo(counting)

    assert reopened[root.definition].definition == root.definition
    assert reopened.get(root.definition).one().definition == root.definition
    assert counting.definition_reads >= 1
    assert counting.definition_enumerations == 0


def test_explicit_weak_cached_child_is_a_retained_get_candidate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = LookupLeaf("child", repo=repo)
    root = LookupRoot(child, repo=repo)
    repo.cache_weak(root)
    repo.cache_weak(child)

    with pytest.raises(RepoLoadError, match="No connected Store"):
        repo.query(child.definition).known().objects()
    with pytest.raises(RepoLoadError, match="No connected Store"):
        repo.get(child.definition)
    with pytest.raises(RepoLoadError, match="No connected Store"):
        repo[child.definition]
    assert repo.get(lambda value: value is child).count() == 0
    repo.cache_strong(child)
    assert repo.get(lambda value: value is child).one() is child


def test_retained_get_preserves_alias_only_root_candidate_policy(tmp_path):
    from dryml.core.store.records import DefinitionRecord, ObjectAliasRecord
    from dryml.core.utils.graph.path import GraphPath

    store = DirStore(tmp_path / "store")
    cdef = Definition(LookupStatefulLeaf, "alias-only").concretize()
    store.write_definition_record(DefinitionRecord(cdef), stored_root=False)
    store.write_object_alias(ObjectAliasRecord("selected", ObjectRef(cdef, {GraphPath(): ObjectId()})))
    repo = Repo(store)

    expected = repo.query(cdef).known().defs().count()
    assert repo.get(cdef).count() == expected
