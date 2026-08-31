import sqlite3

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class ReferenceRebuildValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_old_query_metadata_rebuilds_with_visible_progress(tmp_path, capsys):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    repo.save_object(ReferenceRebuildValue(1, repo=repo))
    index = store.open_query_index()
    index.rebuild()
    con = sqlite3.connect(index.path)
    con.execute("UPDATE catalog_state SET canonical_version = canonical_version - 1")
    con.commit()
    con.close()

    index.rebuild(force=False)
    captured = capsys.readouterr()
    assert "older" in captured.err
    assert "progress" in captured.err
