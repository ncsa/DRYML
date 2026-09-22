import sqlite3

from dryml.core import Definition, Object, ObjectId, ObjectRef, Repo, Serializable
from dryml.core.query.codecs import decode_reference, encode_reference
from dryml.core.query.path import GraphPath
from dryml.core.store.dir import DirStore
from dryml.core.store.records import ObjectAliasRecord


class ReferenceIndexValue(Object):
    def __init__(self, value):
        self.value = value


def test_reference_query_codec_round_trips_complete_values():
    definition = Definition(ReferenceIndexValue, 1).concretize()
    object_ref = ObjectRef(definition, {})
    for value in (ObjectId(("test",)), object_ref):
        assert decode_reference(encode_reference(value)) == value


def test_alias_only_authority_supports_query_index_rebuild(tmp_path):
    """An alias can supply structural authority without a definition record."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    reference = ObjectRef(Definition(ReferenceIndexValue, 1).concretize(), {})
    store.write_object_alias(ObjectAliasRecord("only", reference))

    metadata = store.query_index_record_metadata(reference.definition)
    assert metadata[0] == reference.digest()
    assert metadata[1] == "refs/objects/only.record"
    index = store.open_query_index()
    index.rebuild(force=True)
    assert not store.query_index_is_dirty()
    assert Repo(store).query().stored().one() == reference.definition


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
    index.rebuild()
    con = sqlite3.connect(index.path)
    before = tuple(con.execute("SELECT reference_kind, reference_digest FROM reference_records ORDER BY 1, 2"))
    assert before
    assert con.execute("SELECT COUNT(*) FROM reference_object_ids").fetchone()[0] == 2
    con.close()

    repo.close(flush=True)
    index.path.unlink()
    assert repo.references().object_id(state.object_id).state_refs().one() == state
    con = sqlite3.connect(index.path)
    after = tuple(con.execute("SELECT reference_kind, reference_digest FROM reference_records ORDER BY 1, 2"))
    con.close()
    assert after == before
