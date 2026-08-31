from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from ...cdef_graph import CDEF_GRAPH_SCHEMA_VERSION
from ..codecs import CDEF_CODEC_VERSION, FEATURE_CODEC_VERSION, PATH_CODEC_VERSION, QUERY_INDEX_CODEC_VERSION
from ..model import CANONICAL_QUERY_SEMANTICS_VERSION, FINGERPRINT_SCHEMA_VERSION, QueryIndexDirty, QueryIndexIncompatible


SQLITE_QUERY_INDEX_APPLICATION_ID = 0x44524D4C
SQLITE_QUERY_INDEX_SCHEMA_VERSION = 6
IndexCompatibilityDecision = Literal["compatible", "rebuild", "future-unsupported"]


@dataclass(frozen=True, slots=True)
class IndexSemanticVersion:
    """Semantic version bundle required by a SQLite query index."""

    schema_version: int
    graph_schema_version: int
    path_schema_version: int
    fingerprint_version: int
    cdef_codec_version: int
    feature_codec_version: int
    query_index_codec_version: int
    canonical_version: int
    store_key: str

    def catalog_state(self) -> dict[str, int | str]:
        """Return the subset stored in the SQLite `catalog_state` table."""

        return {
            "schema_version": self.schema_version,
            "graph_schema_version": self.graph_schema_version,
            "path_schema_version": self.path_schema_version,
            "fingerprint_version": self.fingerprint_version,
            "cdef_codec_version": self.cdef_codec_version,
            "feature_codec_version": self.feature_codec_version,
            "query_index_codec_version": self.query_index_codec_version,
            "canonical_version": self.canonical_version,
            "store_key": self.store_key,
        }


DDL = (
    """
    CREATE TABLE IF NOT EXISTS catalog_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        index_uuid TEXT NOT NULL,
        generation INTEGER NOT NULL,
        schema_version INTEGER NOT NULL,
        graph_schema_version INTEGER NOT NULL,
        path_schema_version INTEGER NOT NULL,
        fingerprint_version INTEGER NOT NULL,
        cdef_codec_version INTEGER NOT NULL,
        feature_codec_version INTEGER NOT NULL,
        query_index_codec_version INTEGER NOT NULL,
        canonical_version INTEGER NOT NULL,
        store_key TEXT NOT NULL,
        build_state TEXT NOT NULL CHECK (build_state IN ('building', 'ready', 'dirty')),
        dirty INTEGER NOT NULL CHECK (dirty IN (0, 1)),
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
        unordered INTEGER NOT NULL DEFAULT 0 CHECK (unordered IN (0, 1)),
        edge_kind TEXT NOT NULL DEFAULT 'materialize',
        child_def_id INTEGER NOT NULL REFERENCES definitions(def_id) ON DELETE CASCADE,
        PRIMARY KEY (parent_def_id, edge_kind, path_hash, path_blob, unordered, child_def_id)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS definition_edges_by_child ON definition_edges(child_def_id, edge_kind, path_hash, parent_def_id)",
    "CREATE INDEX IF NOT EXISTS definition_edges_by_parent_path ON definition_edges(parent_def_id, edge_kind, path_hash, child_def_id)",
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
    """
    CREATE TABLE IF NOT EXISTS reference_records (
        source_kind TEXT NOT NULL,
        source_digest TEXT NOT NULL,
        owner_kind TEXT NOT NULL,
        owner_digest TEXT NOT NULL,
        path_blob BLOB NOT NULL,
        reference_kind TEXT NOT NULL CHECK (reference_kind IN ('object', 'state')),
        reference_digest TEXT NOT NULL,
        reference_blob BLOB NOT NULL,
        state_hashes_blob BLOB,
        alias TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (source_kind, source_digest, owner_kind, owner_digest, path_blob, reference_kind, reference_digest, alias)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS reference_records_by_reference ON reference_records(reference_kind, reference_digest)",
    "CREATE INDEX IF NOT EXISTS reference_records_by_alias ON reference_records(alias) WHERE alias <> ''",
    """
    CREATE TABLE IF NOT EXISTS reference_object_ids (
        reference_kind TEXT NOT NULL CHECK (reference_kind IN ('object', 'state')),
        reference_digest TEXT NOT NULL,
        object_id_blob BLOB NOT NULL,
        namespace_blob BLOB NOT NULL,
        path_blob BLOB NOT NULL,
        state_hash TEXT,
        PRIMARY KEY (reference_kind, reference_digest, object_id_blob, path_blob)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS reference_object_ids_by_object ON reference_object_ids(object_id_blob)",
    "CREATE INDEX IF NOT EXISTS reference_object_ids_by_state ON reference_object_ids(state_hash) WHERE state_hash IS NOT NULL",
)


def initialize_schema(con, *, store_key: str, canonical_version: int = CANONICAL_QUERY_SEMANTICS_VERSION, build_state: str = "ready") -> None:
    if build_state not in {"building", "ready", "dirty"}:
        raise ValueError("build_state must be 'building', 'ready', or 'dirty'.")
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if tables:
        validate_schema(con, store_key=store_key, canonical_version=canonical_version, require_ready=False)
        return
    application_id = con.execute("PRAGMA application_id").fetchone()[0]
    user_version = con.execute("PRAGMA user_version").fetchone()[0]
    if application_id != 0 or user_version != 0:
        raise QueryIndexIncompatible("SQLite file is not an empty query-index sidecar.")
    con.execute(f"PRAGMA application_id = {SQLITE_QUERY_INDEX_APPLICATION_ID}")
    con.execute(f"PRAGMA user_version = {SQLITE_QUERY_INDEX_SCHEMA_VERSION}")
    for statement in DDL:
        con.execute(statement)
    _ensure_catalog_state(con, store_key=store_key, canonical_version=canonical_version, build_state=build_state)


def validate_schema(con, *, store_key: str, canonical_version: int = CANONICAL_QUERY_SEMANTICS_VERSION, require_ready: bool = True) -> None:
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
    decision = compatibility_decision(state, expected=expected_semantic_version(
        store_key=store_key,
        canonical_version=canonical_version,
    ))
    if decision != "compatible":
        raise QueryIndexIncompatible(
            f"SQLite query index metadata is {decision}: "
            f"{_compatibility_detail(state, expected)}."
        )
    if require_ready and (state["build_state"] != "ready" or state["dirty"]):
        raise QueryIndexDirty(f"SQLite query index build_state={state['build_state']!r} is not ready.")


def expected_semantic_version(*, store_key: str, canonical_version: int = CANONICAL_QUERY_SEMANTICS_VERSION) -> IndexSemanticVersion:
    """Return the semantic version bundle expected for a Store index."""

    return IndexSemanticVersion(
        schema_version=SQLITE_QUERY_INDEX_SCHEMA_VERSION,
        graph_schema_version=CDEF_GRAPH_SCHEMA_VERSION,
        path_schema_version=PATH_CODEC_VERSION,
        fingerprint_version=FINGERPRINT_SCHEMA_VERSION,
        cdef_codec_version=CDEF_CODEC_VERSION,
        feature_codec_version=FEATURE_CODEC_VERSION,
        query_index_codec_version=QUERY_INDEX_CODEC_VERSION,
        canonical_version=canonical_version,
        store_key=store_key,
    )


def compatibility_decision(
        actual: Mapping[str, int | str],
        *,
        expected: IndexSemanticVersion) -> IndexCompatibilityDecision:
    """Classify how to handle persisted query-index version metadata."""

    for key, value in expected.catalog_state().items():
        actual_value = actual.get(key)
        if key == "store_key":
            if actual_value != value:
                return "rebuild"
            continue
        if type(actual_value) is not int:
            return "future-unsupported"
        if actual_value > value:
            return "future-unsupported"
        if actual_value < value:
            return "rebuild"
    return "compatible"


def stored_compatibility_decision(
        con,
        *,
        store_key: str,
        canonical_version: int = CANONICAL_QUERY_SEMANTICS_VERSION) -> IndexCompatibilityDecision:
    """Classify sidecar metadata without decoding any index rows.

    Args:
        con: Open SQLite connection for the sidecar being inspected.
        store_key: Expected owning Store identifier.
        canonical_version: Expected canonical query-value semantic version.

    Returns:
        ``"compatible"``, ``"rebuild"`` for known older metadata, or
        ``"future-unsupported"`` for future, missing, or malformed metadata.
    """

    application_id = con.execute("PRAGMA application_id").fetchone()[0]
    if application_id != SQLITE_QUERY_INDEX_APPLICATION_ID:
        return "future-unsupported"
    user_version = con.execute("PRAGMA user_version").fetchone()[0]
    if type(user_version) is not int or user_version > SQLITE_QUERY_INDEX_SCHEMA_VERSION:
        return "future-unsupported"
    if user_version < SQLITE_QUERY_INDEX_SCHEMA_VERSION:
        return "rebuild"
    column_names = [info[1] for info in con.execute("PRAGMA table_info(catalog_state)")]
    expected = expected_semantic_version(store_key=store_key, canonical_version=canonical_version)
    if not set(expected.catalog_state()) <= set(column_names):
        return "future-unsupported"
    row = con.execute("SELECT * FROM catalog_state WHERE singleton = 1").fetchone()
    if row is None:
        return "future-unsupported"
    state = dict(zip(column_names, row))
    return compatibility_decision(state, expected=expected)


def _ensure_catalog_state(con, *, store_key: str, canonical_version: int, build_state: str) -> None:
    existing = con.execute("SELECT 1 FROM catalog_state WHERE singleton = 1").fetchone()
    if existing is not None:
        validate_schema(con, store_key=store_key, canonical_version=canonical_version, require_ready=False)
        return
    now = datetime.now(timezone.utc).isoformat()
    values = _semantic_versions(store_key=store_key, canonical_version=canonical_version)
    con.execute(
        """
        INSERT INTO catalog_state (
            singleton,
            index_uuid,
            generation,
            schema_version,
            graph_schema_version,
            path_schema_version,
            fingerprint_version,
            cdef_codec_version,
            feature_codec_version,
            query_index_codec_version,
            canonical_version,
            store_key,
            build_state,
            dirty,
            created_at,
            updated_at
        ) VALUES (1, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            values["schema_version"],
            values["graph_schema_version"],
            values["path_schema_version"],
            values["fingerprint_version"],
            values["cdef_codec_version"],
            values["feature_codec_version"],
            values["query_index_codec_version"],
            values["canonical_version"],
            values["store_key"],
            build_state,
            1 if build_state == "dirty" else 0,
            now,
            now,
        ),
    )


def _semantic_versions(*, store_key: str, canonical_version: int) -> dict[str, int | str]:
    return expected_semantic_version(store_key=store_key, canonical_version=canonical_version).catalog_state()


def _compatibility_detail(actual: Mapping[str, int | str], expected: Mapping[str, int | str]) -> str:
    for key, value in expected.items():
        if actual.get(key) != value:
            return f"{key}={actual.get(key)!r}, expected {value!r}"
    return "unknown metadata mismatch"
