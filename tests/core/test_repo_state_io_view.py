from pathlib import Path

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class ViewValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def test_state_io_view_does_not_open_or_commit_borrowed_store_indexes(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    caller = Repo(store)
    view = Repo._for_state_io((store,))
    opened = []
    committed = []
    monkeypatch.setattr(store, "open_query_index", lambda: opened.append(True))
    monkeypatch.setattr(store, "commit", lambda: committed.append(True))

    obj = ViewValue(4, repo=view)
    state = view.save_object(obj)
    obj.value = 7
    view.restore_state_ref_into(obj, state)
    view.close()

    assert obj.value == 4
    assert opened == []
    assert committed == []
    assert caller.default_store is store


def test_save_releases_its_graph_token_when_store_setup_fails(tmp_path, monkeypatch):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ViewValue(4, repo=repo)
    original = repo._ensure_store

    monkeypatch.setattr(
        repo, "_ensure_store",
        lambda store: (_ for _ in ()).throw(RuntimeError("store setup failed")),
    )
    try:
        try:
            repo.save_object(obj)
        except RuntimeError as error:
            assert str(error) == "store setup failed"
    finally:
        monkeypatch.setattr(repo, "_ensure_store", original)

    with repo.reserve_state_graph(obj):
        pass


def test_state_io_view_preserves_an_active_caller_sqlite_connection(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    caller = Repo(store)
    index = store.open_query_index()
    connection = index._connections.connection()
    view = Repo._for_state_io((store,))
    obj = ViewValue(4, repo=view)
    state = view.save_object(obj)
    obj.value = 7
    view.restore_state_ref_into(obj, state)
    view.close()

    assert store.open_query_index() is index
    assert connection.execute("SELECT 1").fetchone() == (1,)
    assert obj.value == 4
    assert caller.default_store is store
