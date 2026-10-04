import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo
from dryml.core.store.dir import DirStore


class RefreshLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


def test_v3_query_observes_committed_external_writes(tmp_path):
    store = DirStore(tmp_path / "store")
    repo_a = Repo(stores=store)
    first = RefreshLeaf("first", repo=repo_a)
    repo_a.save_object(first)

    repo_view = Repo(stores=DirStore(store.base_dir))
    assert repo_view.query().cdefs().stored().count() == 1

    repo_b = Repo(stores=DirStore(store.base_dir))
    second = RefreshLeaf("second", repo=repo_b)
    repo_b.save_object(second)

    assert repo_view.query().cdefs().stored().count() == 2


def test_current_process_save_updates_v3_cached_and_stored_membership(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = RefreshLeaf("first", repo=repo)
    repo.add_objects(first)

    assert repo.query().cdefs().cached(scope=repo).count() == 1
    assert repo.query().cdefs().stored().count() == 0

    repo.save_object(first)

    assert repo.query().cdefs().stored().count() == 1
