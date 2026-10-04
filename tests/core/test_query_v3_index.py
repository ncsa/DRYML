"""Focused U6 contracts for derived Query V3 index projections.

The projection ledger is deliberately advisory.  These tests assert its useful
integrity boundary without treating a ready sidecar as complete Store authority.
"""

from __future__ import annotations

import sqlite3

import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.query.memory import MemoryStoreQueryIndex
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.query.sqlite.schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION
from dryml.core.query.source import SourceCapture
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, MainRefRecord, ObjectAliasRecord
from dryml.core.store.zip import ZipStore


class ProjectionLeaf(Object):
    """Small graph leaf used to retain sharing topology in a V3 projection."""

    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class ProjectionPair(Object):
    """Small graph root with two independently inspectable edges."""

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right


def _graph_variants():
    shared_leaf = ProjectionLeaf("same")
    shared = ProjectionPair(shared_leaf, shared_leaf).definition
    separate = ProjectionPair(ProjectionLeaf("same"), ProjectionLeaf("same")).definition
    assert shared == separate
    assert not shared.graph_equal(separate)
    return shared, separate


def _sqlite_index(tmp_path, *, store=None):
    return SQLiteStoreQueryIndex(
        source_key="v3-projection-store",
        path=tmp_path / "v3-projection.sqlite",
        config=SQLiteQueryIndexConfig(journal_mode="delete"),
        store=store,
    )


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_memory_and_sqlite_retain_graph_distinct_projection_roots(tmp_path):
    """Projection rows retain graph topology even when legacy CDefs compare equal."""

    shared, separate = _graph_variants()
    graph = ConcreteDefinitionGraph.from_roots((shared, separate))
    store = DirStore(tmp_path / "memory", query_index="memory")
    memory = MemoryStoreQueryIndex(Repo(store)._query_catalog, store)
    memory.register_stored_roots(graph, (shared, separate))

    sqlite = _sqlite_index(tmp_path)
    sqlite.register_stored_roots(graph, (shared, separate))

    memory_coverage = memory.v3_projection_coverage()
    sqlite_coverage = sqlite.v3_projection_coverage()
    assert memory_coverage.source_roots == 2
    assert sqlite_coverage.source_roots == 2
    assert sqlite_coverage.identities >= memory_coverage.identities
    assert memory_coverage.projection_complete
    assert sqlite_coverage.projection_complete
    assert not memory_coverage.indexed_only
    assert not sqlite_coverage.indexed_only


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
@pytest.mark.parametrize("family", ("identity", "relationship"))
def test_missing_v3_projection_row_disables_indexed_only_claim(tmp_path, family):
    """A ready legacy sidecar cannot turn a deleted candidate row into absence."""

    shared, separate = _graph_variants()
    index = _sqlite_index(tmp_path)
    index.register_stored_roots(
        ConcreteDefinitionGraph.from_roots((shared, separate)), (shared, separate),
    )
    assert index.v3_projection_coverage().projection_complete

    with sqlite3.connect(index.path) as con:
        table = "v3_identity_projection" if family == "identity" else "v3_relationship_projection"
        con.execute(
            f"DELETE FROM {table} WHERE root_graph_hash = "
            f"(SELECT root_graph_hash FROM {table} LIMIT 1)"
        )

    coverage = index.v3_projection_coverage()
    assert not coverage.projection_complete
    assert not coverage.authority_complete
    assert not coverage.indexed_only


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_v3_schema_mismatch_rebuilds_only_the_sidecar(tmp_path):
    """Version-7 sidecars rebuild derived rows without touching Store records."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    root = Definition(ProjectionLeaf, "stored").concretize()
    record = store.write_definition_record(DefinitionRecord(root))
    index = store.open_query_index()
    index.rebuild()
    before = store.read_definition_record(record.digest).to_bytes()
    with sqlite3.connect(index.path) as con:
        con.execute("PRAGMA user_version = 7")

    index.refresh("auto")

    with sqlite3.connect(index.path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SQLITE_QUERY_INDEX_SCHEMA_VERSION
    assert store.read_definition_record(record.digest).to_bytes() == before
    assert index.v3_projection_coverage().projection_complete


def test_no_index_and_zip_capture_follow_current_main_authority(tmp_path):
    """Main and alias movement are observed from authority, not index flags."""

    first = Definition(ProjectionLeaf, "first").concretize()
    second = Definition(ProjectionLeaf, "second").concretize()
    directory = DirStore(tmp_path / "directory", query_index="none")
    directory.write_definition_record(DefinitionRecord(first))
    directory.write_definition_record(DefinitionRecord(second))
    directory.write_main_ref(MainRefRecord(DefinitionRecord(first).digest))
    first_facts = SourceCapture().capture_store(directory)
    directory.write_main_ref(MainRefRecord(DefinitionRecord(second).digest))
    second_facts = SourceCapture().capture_store(directory)
    assert first_facts.main_definition == first
    assert second_facts.main_definition == second

    repo = Repo(directory)
    first_state = repo.save_object(ProjectionLeaf("alias-first", repo=repo))
    second_state = repo.save_object(ProjectionLeaf("alias-second", repo=repo))
    directory.write_object_alias(ObjectAliasRecord("current", first_state.object))
    first_alias_facts = SourceCapture().capture_store(directory)
    directory.write_object_alias(ObjectAliasRecord("current", second_state.object))
    second_alias_facts = SourceCapture().capture_store(directory)
    assert first_alias_facts.object_aliases[0].object_ref == first_state.object
    assert second_alias_facts.object_aliases[0].object_ref == second_state.object

    archive = tmp_path / "authority.zip"
    seed = ZipStore(archive)
    seed.write_definition_record(DefinitionRecord(second))
    seed.write_main_ref(MainRefRecord(DefinitionRecord(second).digest))
    seed.commit()
    seed.close()
    zipped = ZipStore.open_existing(archive)
    try:
        assert second in SourceCapture().capture_store(zipped).knowledge()
    finally:
        zipped.close()


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_incremental_v3_publication_does_not_rescan_historical_projection_rows(tmp_path, monkeypatch):
    """A clean save updates coverage with its delta, not the entire sidecar."""

    from dryml.core.query.sqlite import index as sqlite_index

    index = _sqlite_index(tmp_path)
    first = Definition(ProjectionLeaf, "first").concretize()
    second = Definition(ProjectionLeaf, "second").concretize()
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(first), (first,))
    monkeypatch.setattr(
        sqlite_index, "_v3_projection_digest",
        lambda *_args, **_kwargs: pytest.fail("incremental save scanned historical projection rows"),
    )

    index.register_stored_roots(ConcreteDefinitionGraph.from_root(second), (second,))
    index.remove_stored_roots((first,))
