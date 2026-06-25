from __future__ import annotations

from datetime import datetime, timezone

from ...cdef_graph import CDEF_GRAPH_SCHEMA_VERSION
from ..codecs import CDEF_CODEC_VERSION, FEATURE_CODEC_VERSION, PATH_CODEC_VERSION
from ..model import FINGERPRINT_SCHEMA_VERSION, QueryIndexIncompatible


SQLITE_QUERY_INDEX_APPLICATION_ID = 0x44524D4C
SQLITE_QUERY_INDEX_SCHEMA_VERSION = 1


DDL = (
    """
    CREATE TABLE IF NOT EXISTS catalog_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        generation INTEGER NOT NULL,
        schema_version INTEGER NOT NULL,
        graph_schema_version INTEGER NOT NULL,
        path_schema_version INTEGER NOT NULL,
        fingerprint_version INTEGER NOT NULL,
        cdef_codec_version INTEGER NOT NULL,
        feature_codec_version INTEGER NOT NULL,
        canonical_version INTEGER NOT NULL,
        store_key TEXT NOT NULL,
        build_state TEXT NOT NULL CHECK (build_state IN ('building', 'ready', 'dirty')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS definitions (
        def_id INTEGER PRIMARY KEY,
        stable_hash BLOB NOT NULL,
        collision_ordinal INTEGER NOT NULL,
        class_key BLOB NOT NULL,
        cdef_blob BLOB NOT NULL,
        created_generation INTEGER NOT NULL,
        UNIQUE (stable_hash, collision_ordinal)
    )
    """,
    "CREATE INDEX IF NOT EXISTS definitions_by_stable_hash ON definitions(stable_hash)",
    "CREATE INDEX IF NOT EXISTS definitions_by_class_key ON definitions(class_key, def_id)",
    """
    CREATE TABLE IF NOT EXISTS feature_tokens (
        feature_id INTEGER PRIMARY KEY,
        token_hash BLOB NOT NULL,
        collision_ordinal INTEGER NOT NULL,
        token_blob BLOB NOT NULL,
        document_frequency INTEGER NOT NULL DEFAULT 0,
        UNIQUE (token_hash, collision_ordinal)
    )
    """,
    "CREATE INDEX IF NOT EXISTS feature_tokens_by_hash ON feature_tokens(token_hash)",
    """
    CREATE TABLE IF NOT EXISTS postings (
        feature_id INTEGER NOT NULL REFERENCES feature_tokens(feature_id) ON DELETE CASCADE,
        def_id INTEGER NOT NULL REFERENCES definitions(def_id) ON DELETE CASCADE,
        multiplicity INTEGER NOT NULL CHECK (multiplicity > 0),
        PRIMARY KEY (feature_id, def_id)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS postings_by_definition ON postings(def_id, feature_id)",
    """
    CREATE TABLE IF NOT EXISTS definition_edges (
        parent_def_id INTEGER NOT NULL REFERENCES definitions(def_id) ON DELETE CASCADE,
        path_hash BLOB NOT NULL,
        path_blob BLOB NOT NULL,
        child_def_id INTEGER NOT NULL REFERENCES definitions(def_id) ON DELETE CASCADE,
        PRIMARY KEY (parent_def_id, path_hash, path_blob, child_def_id)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS definition_edges_by_child ON definition_edges(child_def_id, path_hash, parent_def_id)",
    "CREATE INDEX IF NOT EXISTS definition_edges_by_parent_path ON definition_edges(parent_def_id, path_hash, child_def_id)",
    """
    CREATE TABLE IF NOT EXISTS stored_roots (
        def_id INTEGER PRIMARY KEY REFERENCES definitions(def_id) ON DELETE CASCADE,
        storage_hash BLOB NOT NULL,
        relative_def_path TEXT NOT NULL,
        def_size INTEGER,
        def_mtime_ns INTEGER,
        indexed_generation INTEGER NOT NULL
    )
    """,
)


def initialize_schema(con, *, store_key: str, canonical_version: int = 1) -> None:
    con.execute(f"PRAGMA application_id = {SQLITE_QUERY_INDEX_APPLICATION_ID}")
    con.execute(f"PRAGMA user_version = {SQLITE_QUERY_INDEX_SCHEMA_VERSION}")
    for statement in DDL:
        con.execute(statement)
    _ensure_catalog_state(con, store_key=store_key, canonical_version=canonical_version)


def validate_schema(con, *, store_key: str, canonical_version: int = 1) -> None:
    application_id = con.execute("PRAGMA application_id").fetchone()[0]
    if application_id != SQLITE_QUERY_INDEX_APPLICATION_ID:
        raise QueryIndexIncompatible("SQLite file is not a DRYML query index.")
    user_version = con.execute("PRAGMA user_version").fetchone()[0]
    if user_version != SQLITE_QUERY_INDEX_SCHEMA_VERSION:
        raise QueryIndexIncompatible(f"Unsupported SQLite query-index schema version {user_version!r}.")
    row = con.execute("SELECT * FROM catalog_state WHERE singleton = 1").fetchone()
    if row is None:
        raise QueryIndexIncompatible("SQLite query index is missing catalog_state.")
    columns = [info[1] for info in con.execute("PRAGMA table_info(catalog_state)")]
    state = dict(zip(columns, row))
    expected = _semantic_versions(store_key=store_key, canonical_version=canonical_version)
    for key, value in expected.items():
        if state[key] != value:
            raise QueryIndexIncompatible(f"SQLite query index {key}={state[key]!r} is incompatible with expected {value!r}.")


def _ensure_catalog_state(con, *, store_key: str, canonical_version: int) -> None:
    existing = con.execute("SELECT 1 FROM catalog_state WHERE singleton = 1").fetchone()
    if existing is not None:
        validate_schema(con, store_key=store_key, canonical_version=canonical_version)
        return
    now = datetime.now(timezone.utc).isoformat()
    values = _semantic_versions(store_key=store_key, canonical_version=canonical_version)
    con.execute(
        """
        INSERT INTO catalog_state (
            singleton,
            generation,
            schema_version,
            graph_schema_version,
            path_schema_version,
            fingerprint_version,
            cdef_codec_version,
            feature_codec_version,
            canonical_version,
            store_key,
            build_state,
            created_at,
            updated_at
        ) VALUES (1, 0, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
        """,
        (
            values["schema_version"],
            values["graph_schema_version"],
            values["path_schema_version"],
            values["fingerprint_version"],
            values["cdef_codec_version"],
            values["feature_codec_version"],
            values["canonical_version"],
            values["store_key"],
            now,
            now,
        ),
    )


def _semantic_versions(*, store_key: str, canonical_version: int) -> dict[str, int | str]:
    return {
        "schema_version": SQLITE_QUERY_INDEX_SCHEMA_VERSION,
        "graph_schema_version": CDEF_GRAPH_SCHEMA_VERSION,
        "path_schema_version": PATH_CODEC_VERSION,
        "fingerprint_version": FINGERPRINT_SCHEMA_VERSION,
        "cdef_codec_version": CDEF_CODEC_VERSION,
        "feature_codec_version": FEATURE_CODEC_VERSION,
        "canonical_version": canonical_version,
        "store_key": store_key,
    }
