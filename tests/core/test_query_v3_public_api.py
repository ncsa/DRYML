"""Characterization tests for the retained non-query Repo lookup surface.

Query V3 cutover must not change item lookup, ``get()``, or explicit load
semantics while public query producers migrate separately.
"""

import pytest

from dryml.core import Definition, Object, Repo, SKIP_ARGS
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


def test_retained_item_get_and_load_lookups_keep_their_candidate_universes(tmp_path):
    """Item/get retain known roots and cache entries; explicit loads stay structural."""

    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    child = LookupLeaf("child", repo=writer)
    root = LookupRoot(child, repo=writer)
    writer.save_object(root)

    reopened = Repo(DirStore(store.base_dir))

    assert reopened[root.definition].definition == root.definition
    assert reopened.get(root.definition).one().definition == root.definition
    assert reopened.get(child.definition).count() == 0
    with pytest.raises(KeyError):
        reopened[child.definition]
    with pytest.raises(TypeError, match="ConcreteDefinition"):
        reopened[Definition(LookupRoot, SKIP_ARGS)]

    # The child definition record is structural authority but not a get/item
    # candidate until it is an explicitly retained cache member.
    assert reopened.load_object(child.definition).definition == child.definition
    missing = LookupLeaf("missing", repo=reopened).definition
    with pytest.raises(RepoLoadError, match="No connected Store"):
        reopened.load_object(missing)

    reopened.cache_strong(child)
    assert reopened.get(child.definition).one() is child
    assert reopened.get(lambda value: value is child).one() is child
