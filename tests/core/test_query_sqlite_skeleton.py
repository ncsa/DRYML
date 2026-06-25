import pytest

from dryml.core2.query.model import QueryIndexIncompatible
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
from dryml.core2.query.sqlite.connection import SQLiteConnectionManager
from dryml.core2.query.sqlite.schema import (
    SQLITE_QUERY_INDEX_APPLICATION_ID,
    SQLITE_QUERY_INDEX_SCHEMA_VERSION,
    initialize_schema,
    validate_schema,
)
from dryml.core2.query.sqlite.utils import is_sqlite_busy_error, wal_runtime_is_known_safe


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
    manager = SQLiteConnectionManager(SQLiteQueryIndexConfig(tmp_path / "index.sqlite", journal_mode="delete"))
    first = manager.connection()
    second = manager.connection()

    assert first is second
    assert first.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    manager.close_current()
    third = manager.connection()
    assert third is not first
    manager.close_all_current_process()


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
