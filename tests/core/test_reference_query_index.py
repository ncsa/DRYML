import sqlite3

from dryml.core import Definition, Object, ObjectId, ObjectRef, Repo, Serializable
from dryml.core.query.codecs import decode_reference, encode_reference
from dryml.core.query.path import GraphPath
from dryml.core.store.dir import DirStore


class ReferenceIndexValue(Object):
    def __init__(self, value):
        self.value = value


def test_reference_query_codec_round_trips_complete_values():
    definition = Definition(ReferenceIndexValue, 1).concretize()
    object_ref = ObjectRef(definition, {})
    for value in (ObjectId(("test",)), object_ref):
        assert decode_reference(encode_reference(value)) == value


class IndexedReferenceValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_sqlite_reference_rows_rebuild_from_unchanged_authority(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    state = Repo(store).save_object(IndexedReferenceValue(1))
    repo = Repo(DirStore(store.base_dir, query_index="sqlite"))

    assert repo.references().object_id(state.object_id).state_refs().one() == state
    index = repo.default_store.open_query_index()
    con = sqlite3.connect(index.path)
    before = tuple(con.execute("SELECT reference_kind, reference_digest FROM reference_records ORDER BY 1, 2"))
    assert before
    assert con.execute("SELECT COUNT(*) FROM reference_object_ids").fetchone()[0] == 2
    con.close()

    index.path.unlink()
    assert repo.references().object_id(state.object_id).state_refs().one() == state
    con = sqlite3.connect(index.path)
    after = tuple(con.execute("SELECT reference_kind, reference_digest FROM reference_records ORDER BY 1, 2"))
    con.close()
    assert after == before
