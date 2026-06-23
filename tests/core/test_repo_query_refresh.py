from dryml.core2 import Object, Repo
from dryml.core2.store.dir import DirStore


class RefreshLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


def test_refresh_false_auto_and_forced_refresh_visibility(tmp_path):
    store = DirStore(tmp_path / "store")
    repo_a = Repo(stores=store)
    first = RefreshLeaf("first", repo=repo_a)
    repo_a.save_object(first)

    repo_view = Repo(stores=DirStore(store.base_dir))
    assert len(repo_view.find_defs(None, refresh=False)) == 0
    assert len(repo_view.find_defs(None)) == 1

    repo_b = Repo(stores=DirStore(store.base_dir))
    second = RefreshLeaf("second", repo=repo_b)
    repo_b.save_object(second)

    # First-sprint contract: already-hydrated stores need forced refresh for
    # external writes from another Repo instance.
    assert len(repo_view.find_defs(None)) == 1
    assert len(repo_view.find_defs(None, refresh=True)) == 2


def test_current_process_save_updates_query_catalog_without_refresh(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = RefreshLeaf("first", repo=repo)
    repo.add_objects(first)

    assert len(repo.find_defs(None, scope="cached", refresh=False)) == 1
    assert len(repo.find_defs(None, scope="stored", refresh=False)) == 0

    repo.save_object(first)

    assert len(repo.find_defs(None, scope="stored", refresh=False)) == 1
