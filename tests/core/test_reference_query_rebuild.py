import sqlite3

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Serializable
from dryml.core.store.dir import DirStore


class ReferenceRebuildValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_owned_schema9_index_rebuilds_with_visible_progress(tmp_path, capsys):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(Definition(ReferenceRebuildValue, value=1).concretize())
    index = store.open_query_index()
    index.rebuild()
    con = sqlite3.connect(index.path)
    con.execute("PRAGMA user_version=9")
    con.commit()
    con.close()

    index.rebuild(force=False)
    captured = capsys.readouterr()
    assert "older" in captured.err
    assert "metadata-only rebuild" in captured.err
    assert index.status().schema_version == 11
    assert len(index.projection_rows("recipe_root", refresh=False)) == 1
