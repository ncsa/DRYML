import pytest

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS, Serializable
from dryml.core2.query import QueryDomainError
from dryml.core2.store.dir import DirStore


class QueryLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class QueryParent(Serializable):
    def __init__(self, child, *, label="parent"):
        super().__init__()
        self.child = child
        self.label = label


def test_nested_query_finds_ephemeral_child_and_owner(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("child", repo=repo)
    parent = QueryParent(child, repo=repo)
    repo.save_object(parent)

    assert store.has(parent.definition)
    assert not store.has(child.definition)

    repo2 = Repo(stores=DirStore(store.base_dir))
    selector = Definition(QueryLeaf, SKIP_ARGS)

    assert len(repo2.find_defs(selector, scope="stored")) == 0

    occurrences = repo2.find_occurrences(selector)
    assert occurrences.count() == 1
    occurrence = occurrences.one()
    assert occurrence.owner == parent.definition
    assert occurrence.definition == child.definition
    assert str(occurrence.path) == "$.args[0]"

    owner_defs = repo2.find_owner_defs(selector)
    assert list(owner_defs) == [parent.definition]

    with pytest.raises(QueryDomainError):
        repo2.query(selector).nested().objects()

    owners = repo2.query(selector).nested().owners().objects()
    assert owners.one().child.name == "child"


def test_nested_occurrences_preserve_duplicate_paths(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("shared", repo=repo)
    parent = QueryParent([child, child], repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    occurrences = repo2.find_occurrences(Definition(QueryLeaf, SKIP_ARGS))

    assert occurrences.count() == 2
    assert {str(occ.path) for occ in occurrences} == {"$.args[0][0]", "$.args[0][1]"}
    assert occurrences.definitions().count() == 1
    assert occurrences.owners().count() == 1
