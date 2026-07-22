import os
import threading
from uuid import UUID

import pytest

from dryml.core.query.model import QueryIndexDirty, QueryIndexError, QueryIndexIncompatible
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
import dryml.core.query.sqlite.connection as connection_module
from dryml.core.query.sqlite.connection import SQLiteConnectionManager
from dryml.core.query.sqlite.schema import (
    IndexSemanticVersion,
    SQLITE_QUERY_INDEX_APPLICATION_ID,
    SQLITE_QUERY_INDEX_SCHEMA_VERSION,
    compatibility_decision,
    expected_semantic_version,
    initialize_schema,
    validate_schema,
)
from dryml.core.query.sqlite.utils import is_sqlite_busy_error, wal_runtime_is_known_safe


pytestmark = pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")


def test_wal_runtime_safety_policy():
    assert wal_runtime_is_known_safe((3, 51, 3))
    assert wal_runtime_is_known_safe((3, 50, 7))
    assert wal_runtime_is_known_safe((3, 44, 6))
    assert not wal_runtime_is_known_safe((3, 51, 2))
    assert not wal_runtime_is_known_safe((3, 50, 6))
    assert not wal_runtime_is_known_safe((3, 44, 5))


def test_busy_error_classifier():
    sqlite3 = require_sqlite()

    assert is_sqlite_busy_error(sqlite3.OperationalError("database is locked"))
    assert is_sqlite_busy_error(sqlite3.DatabaseError("database table is locked"))
    assert not is_sqlite_busy_error(sqlite3.OperationalError("no such table: definitions"))
    assert not is_sqlite_busy_error(ValueError("database is locked"))


def test_connection_manager_is_process_thread_local(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete", busy_timeout=1.25))
    first = manager.connection()
    second = manager.connection()

    assert first is second
    assert first.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert first.execute("PRAGMA busy_timeout").fetchone()[0] == 1250

    manager.close_current()
    third = manager.connection()
    assert third is not first
    manager.close_all_current_process()


def test_connection_manager_shares_same_thread_connections_until_all_owners_close(tmp_path):
    config = SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete")
    left = SQLiteConnectionManager(config)
    right = SQLiteConnectionManager(config)
    sqlite3 = require_sqlite()

    con = left.connection()
    assert right.connection() is con

    left.close_current()
    assert con.execute("SELECT 1").fetchone()[0] == 1
    right.close_current()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        con.execute("SELECT 1")


def test_connection_manager_active_lease_survives_owner_closes(tmp_path):
    config = SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete")
    left = SQLiteConnectionManager(config)
    right = SQLiteConnectionManager(config)
    sqlite3 = require_sqlite()

    with left.lease() as con:
        assert right.connection() is con
        left.close_current()
        right.close_current()
        assert con.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        con.execute("SELECT 1")


def test_connection_manager_reopens_externally_closed_cached_connection(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()
    con.close()

    reopened = manager.connection()

    assert reopened is not con
    assert reopened.execute("SELECT 1").fetchone()[0] == 1
    manager.close_current()


@pytest.mark.skipif(os.name == "nt", reason="Windows does not replace an open SQLite database")
def test_connection_manager_reopens_when_database_path_identity_changes(tmp_path):
    path = tmp_path / "index.sqlite"
    replacement_path = tmp_path / "replacement.sqlite"
    config = SQLiteQueryIndexConfig(path, journal_mode="delete")
    left = SQLiteConnectionManager(config)
    right = SQLiteConnectionManager(config)
    con = left.connection()
    assert right.connection() is con
    con.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    con.execute("INSERT INTO marker VALUES ('original')")

    sqlite3 = require_sqlite()
    with sqlite3.connect(replacement_path) as replacement:
        replacement.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        replacement.execute("INSERT INTO marker VALUES ('replacement')")
    os.replace(replacement_path, path)

    reopened = left.connection()

    assert reopened is not con
    assert right.connection() is reopened
    assert reopened.execute("SELECT value FROM marker").fetchone()[0] == "replacement"
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        con.execute("SELECT 1")
    left.close_current()
    assert reopened.execute("SELECT 1").fetchone()[0] == 1
    right.close_current()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        reopened.execute("SELECT 1")


@pytest.mark.skipif(os.name == "nt", reason="Windows does not replace an open SQLite database")
def test_connection_manager_rejects_identity_change_during_active_lease(tmp_path):
    path = tmp_path / "index.sqlite"
    replacement_path = tmp_path / "replacement.sqlite"
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(path, journal_mode="delete"))
    sqlite3 = require_sqlite()

    with manager.lease() as con:
        con.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        con.execute("INSERT INTO marker VALUES ('original')")
        with sqlite3.connect(replacement_path) as replacement:
            replacement.execute("CREATE TABLE marker (value TEXT NOT NULL)")
            replacement.execute("INSERT INTO marker VALUES ('replacement')")
        os.replace(replacement_path, path)

        with pytest.raises(QueryIndexError, match="actively leased"):
            manager.connection()
        assert con.execute("SELECT value FROM marker").fetchone()[0] == "original"

    reopened = manager.connection()
    assert reopened is not con
    assert reopened.execute("SELECT value FROM marker").fetchone()[0] == "replacement"
    manager.close_current()


def test_connection_manager_path_invalidation_reopens_other_manager(tmp_path):
    config = SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete")
    left = SQLiteConnectionManager(config)
    right = SQLiteConnectionManager(config)
    sqlite3 = require_sqlite()

    con = left.connection()
    assert right.connection() is con
    right.close_path_current_process()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        con.execute("SELECT 1")

    reopened = left.connection()
    assert reopened is not con
    assert reopened.execute("SELECT 1").fetchone()[0] == 1
    left.close_current()


def test_connection_is_opened_lazily(tmp_path):
    path = tmp_path / "index.sqlite"
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(path, journal_mode="delete"))

    assert not path.exists()

    manager.connection()

    assert path.exists()
    manager.close_all_current_process()


def test_different_thread_uses_different_connection(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    main_con = manager.connection()
    results = []

    def open_in_thread():
        thread_con = manager.connection()
        try:
            results.append((thread_con is main_con, thread_con.execute("PRAGMA foreign_keys").fetchone()[0]))
        finally:
            manager.close_current()

    thread = threading.Thread(target=open_in_thread)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert results == [(False, 1)]
    manager.close_current()


def test_auto_journal_mode_uses_delete_when_wal_runtime_is_not_known_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(connection_module, "wal_runtime_is_known_safe", lambda version: False)
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="auto"))

    con = manager.connection()

    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    manager.close_all_current_process()


def test_durability_setting_applies_synchronous_pragma(tmp_path):
    normal = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "normal.sqlite", journal_mode="delete", durability="normal"))
    full = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "full.sqlite", journal_mode="delete", durability="full"))

    try:
        assert normal.connection().execute("PRAGMA synchronous").fetchone()[0] == 1
        assert full.connection().execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        normal.close_all_current_process()
        full.close_all_current_process()


def test_schema_initialization_and_validation(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")
    validate_schema(con, store_key="store-a")

    assert con.execute("PRAGMA application_id").fetchone()[0] == SQLITE_QUERY_INDEX_APPLICATION_ID
    assert con.execute("PRAGMA user_version").fetchone()[0] == SQLITE_QUERY_INDEX_SCHEMA_VERSION
    state = con.execute("SELECT generation, build_state, store_key FROM catalog_state").fetchone()
    assert state == (0, "ready", "store-a")
    for table in {"definitions", "feature_tokens", "postings", "definition_edges", "stored_roots"}:
        assert con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    manager.close_all_current_process()


def test_catalog_state_persists_required_metadata(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")

    row = con.execute(
        """
        SELECT index_uuid, generation, schema_version, graph_schema_version,
               path_schema_version, fingerprint_version, cdef_codec_version,
               feature_codec_version, canonical_version, store_key, build_state,
               dirty, created_at, updated_at
        FROM catalog_state
        WHERE singleton = 1
        """
    ).fetchone()
    assert row is not None
    UUID(row[0])
    assert row[1] == 0
    assert row[2] == SQLITE_QUERY_INDEX_SCHEMA_VERSION
    assert row[3:9] == tuple(expected_semantic_version(store_key="store-a").catalog_state()[key] for key in (
        "graph_schema_version",
        "path_schema_version",
        "fingerprint_version",
        "cdef_codec_version",
        "feature_codec_version",
        "canonical_version",
    ))
    assert row[9] == "store-a"
    assert row[10] == "ready"
    assert row[11] == 0
    assert row[12]
    assert row[13]

    initialize_schema(con, store_key="store-a")
    assert con.execute("SELECT index_uuid FROM catalog_state WHERE singleton = 1").fetchone()[0] == row[0]
    manager.close_all_current_process()


def test_schema_tables_expose_required_columns_and_constraints(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")

    table_columns = {
        table: {row[1]: row for row in con.execute(f"PRAGMA table_info({table})")}
        for table in ("catalog_state", "definitions", "feature_tokens", "postings", "definition_edges", "stored_roots")
    }
    assert set(table_columns["catalog_state"]) >= {
        "singleton",
        "index_uuid",
        "generation",
        "schema_version",
        "graph_schema_version",
        "fingerprint_version",
        "path_schema_version",
        "cdef_codec_version",
        "feature_codec_version",
        "canonical_version",
        "store_key",
        "build_state",
        "dirty",
        "created_at",
        "updated_at",
    }
    assert set(table_columns["definitions"]) >= {
        "def_id",
        "stable_hash",
        "collision_ordinal",
        "class_key",
        "cdef_blob",
        "created_generation",
    }
    assert set(table_columns["feature_tokens"]) >= {
        "feature_id",
        "token_hash",
        "collision_ordinal",
        "token_blob",
        "document_frequency",
    }
    assert set(table_columns["postings"]) >= {"feature_id", "def_id", "multiplicity"}
    assert set(table_columns["definition_edges"]) >= {
        "parent_def_id",
        "child_def_id",
        "path_hash",
        "path_blob",
        "unordered",
    }
    assert set(table_columns["stored_roots"]) >= {
        "def_id",
        "storage_hash",
        "relative_def_path",
        "def_size",
        "def_mtime_ns",
        "indexed_generation",
    }

    index_columns = {
        row[2]
        for row in con.execute("PRAGMA index_info(postings_by_definition)")
    }
    assert index_columns == {"def_id", "feature_id"}
    edge_indexes = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'definition_edges'")
        if row[0] is not None
    }
    assert {"definition_edges_by_child", "definition_edges_by_parent_path"} <= edge_indexes
    manager.close_all_current_process()


def test_index_semantic_version_and_compatibility_decision():
    expected = expected_semantic_version(store_key="store-a", canonical_version=1)
    actual = expected.catalog_state()

    assert isinstance(expected, IndexSemanticVersion)
    assert compatibility_decision(actual, expected=expected) == "compatible"

    needs_migration = dict(actual)
    needs_migration["schema_version"] = SQLITE_QUERY_INDEX_SCHEMA_VERSION - 1
    assert compatibility_decision(needs_migration, expected=expected) == "migrate"

    future = dict(actual)
    future["schema_version"] = SQLITE_QUERY_INDEX_SCHEMA_VERSION + 1
    assert compatibility_decision(future, expected=expected) == "future-unsupported"


def test_codec_version_mismatch_requests_rebuild():
    expected = expected_semantic_version(store_key="store-a", canonical_version=1)
    actual = expected.catalog_state()
    actual["cdef_codec_version"] = expected.cdef_codec_version - 1

    assert compatibility_decision(actual, expected=expected) == "rebuild"


def test_schema_initialization_is_idempotent_and_has_one_catalog_row(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")
    initialize_schema(con, store_key="store-a")

    assert con.execute("SELECT COUNT(*) FROM catalog_state").fetchone()[0] == 1
    assert con.execute("SELECT generation, build_state, store_key FROM catalog_state").fetchone() == (0, "ready", "store-a")
    manager.close_all_current_process()


def test_required_schema_indexes_exist(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")

    indexes = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        if row[0] is not None
    }
    assert {
        "definitions_by_stable_hash",
        "definitions_by_class_key",
        "feature_tokens_by_hash",
        "postings_by_definition",
        "definition_edges_by_child",
        "definition_edges_by_parent_path",
    } <= indexes
    manager.close_all_current_process()


def test_build_state_blocks_partial_index(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a", build_state="building")

    validate_schema(con, store_key="store-a", require_ready=False)
    with pytest.raises(QueryIndexDirty, match="not ready"):
        validate_schema(con, store_key="store-a")
    manager.close_all_current_process()


def test_schema_rejects_store_key_mismatch(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")
    with pytest.raises(QueryIndexIncompatible):
        validate_schema(con, store_key="store-b")
    manager.close_all_current_process()


def test_schema_rejects_unknown_application_id(tmp_path):
    sqlite3 = require_sqlite()
    con = sqlite3.connect(tmp_path / "not-index.sqlite")

    with pytest.raises(QueryIndexIncompatible):
        validate_schema(con, store_key="store-a")
    con.close()


def test_schema_rejects_future_user_version(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a")
    con.execute(f"PRAGMA user_version = {SQLITE_QUERY_INDEX_SCHEMA_VERSION + 1}")

    with pytest.raises(QueryIndexIncompatible, match="schema version"):
        validate_schema(con, store_key="store-a")
    manager.close_all_current_process()


def test_schema_rejects_semantic_version_mismatch(tmp_path):
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    con = manager.connection()

    initialize_schema(con, store_key="store-a", canonical_version=1)

    with pytest.raises(QueryIndexIncompatible, match="canonical_version"):
        validate_schema(con, store_key="store-a", canonical_version=2)
    manager.close_all_current_process()
