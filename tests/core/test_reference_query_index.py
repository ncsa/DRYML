import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, ObjectId, ObjectRef, Repo, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.query.codecs import decode_reference, encode_reference
from dryml.core.query.path import GraphPath
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.links import DefLink
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


@pytest.mark.parametrize("sidecar_state", ("missing", "dirty", "corrupt"))
def test_reference_containment_uses_current_roots_before_sidecar_recovery(tmp_path, sidecar_state):
    """Root-local containment never substitutes derived membership for authority."""
    store = DirStore(
        tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    repo = Repo(store)
    target = ReferenceIndexValue("contained", repo=repo)
    owner = ReferenceIndexValue(
        DefLink.finalized(EdgeKind.REF, target.definition), repo=repo,
    )
    repo.save_object(owner)
    sidecar = Path(store.query_index_path)
    assert sidecar.exists()

    if sidecar_state == "missing":
        sidecar.unlink()
    elif sidecar_state == "dirty":
        store.mark_query_index_dirty()
    else:
        sidecar.write_bytes(b"not a sqlite database")

    current = DirStore(
        store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    current_repo = Repo(current)
    query = current_repo.query(target.definition).nested(edges="ref", refresh=False).owners()

    assert query.one() == owner.definition
    assert current.query_index_status().state == sidecar_state

    recovered = DirStore(
        store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    assert Repo(recovered).query(target.definition).nested(edges="ref").owners().one() == owner.definition
    assert recovered.query_index_status().state == "ready"


def test_reference_containment_refresh_false_uses_current_roots_not_old_sidecar_membership(tmp_path):
    """A ready but stale sidecar cannot hide an authoritative reference owner."""
    store = DirStore(
        tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    repo = Repo(store)
    target = ReferenceIndexValue("current-root", repo=repo)
    owner = ReferenceIndexValue(
        DefLink.finalized(EdgeKind.REF, target.definition), repo=repo,
    )
    repo.save_object(owner)
    store.open_query_index().remove_stored_roots((owner.definition,))

    current = Repo(DirStore(
        store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    ))

    assert current.query(target.definition).nested(edges="ref", refresh=False).owners().one() == owner.definition
