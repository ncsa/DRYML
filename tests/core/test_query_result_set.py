import pytest

from dryml.core2 import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core2.query import QueryCardinalityError, QueryDomainError
from dryml.core2.store.dir import DirStore


class ResultLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class ResultParent(Serializable):
    def __init__(self, child):
        super().__init__()
        self.child = child


def test_result_set_cardinality_helpers(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    second = ResultLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    many = repo.find_defs(None)
    empty = repo.find_defs(Definition(ResultLeaf, "missing"), refresh=False)
    one = repo.find_defs(Definition(ResultLeaf, "first"), refresh=False)

    assert many.count() == len(many)
    assert many.exists()
    with pytest.raises(QueryCardinalityError):
        many.one()
    with pytest.raises(QueryCardinalityError):
        many.one_or_none()
    with pytest.raises(QueryCardinalityError):
        empty.one()
    assert empty.one_or_none() is None
    assert one.one() == first.definition


def test_refine_preserves_nested_nonmaterializable_domain(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = ResultLeaf("child", repo=repo)
    parent = ResultParent(child, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    nested_defs = repo2.query(Definition(ResultLeaf, SKIP_ARGS)).nested().definitions().defs()
    refined = nested_defs.refine(Definition(ResultLeaf, "child"))

    assert refined.domain == "nested-definitions"
    assert not refined.materializable
    assert list(refined) == [child.definition]
    with pytest.raises(QueryDomainError):
        refined.objects()


def test_result_snapshot_does_not_gain_later_indexed_results(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    repo.save_object(first)
    snapshot = repo.find_defs(None)

    second = ResultLeaf("second", repo=repo)
    repo.save_object(second)

    assert list(snapshot) == [first.definition]
    assert repo.find_defs(None, refresh=False).count() == 2


def test_result_set_refinement_never_expands_original_set(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    second = ResultLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    broad = repo.find_defs(Definition(ResultLeaf, "first"))
    refined = broad.refine(Definition(ResultLeaf, SKIP_ARGS))

    assert list(refined) == [first.definition]


def test_union_and_intersection_are_deterministic_and_deduplicate(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    second = ResultLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    first_rs = repo.find_defs(Definition(ResultLeaf, "first"))
    all_rs = repo.find_defs(Definition(ResultLeaf, SKIP_ARGS))

    assert first_rs.union(first_rs).count() == 1
    assert list(all_rs.intersection(first_rs)) == [first.definition]
