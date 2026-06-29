# Query Index Backend Contracts

Status: draft.

DRYML query indexes are acceleration metadata for `Repo.query(...)`. Backends must preserve exact `ConcreteDefinition` semantics and must not expose backend-local IDs outside a read view.

## Components

`RepoQueryIndex` federates query sources:

- one source per Store
- the live in-memory cache overlay
- generation metadata for each participating source
- Store-priority replica ordering

`StoreQueryIndex` is the Store-owned backend contract. A persistent backend such as `SQLiteStoreQueryIndex` owns sidecar files, connections, write transactions, status, validation, reconciliation, and rebuild.

`QueryIndexReadView` is a short-lived read transaction. It returns Store-local definition IDs, candidate sets, owner projections, and occurrence traversal snapshots for one source generation.

SQLite read views also expose a lowering-capable path for safe SQL-native candidate relations. See `docs/sqlite_lowering.md` for the lowered relation, terminal, scan-policy, diagnostics, and fallback boundaries.

Lowered candidate relations are backend-owned. Federation may inspect only their source key, generation, opaque relation ID, relation kind, ordering contract, estimated rows, exact-safe flag, debug label, and opaque keyset cursor. SQLite may represent the relation as SQL text, CTEs, or temp tables. Active read views can page an owned `CandidateRelation` and compose it with backend methods for domain filtering, parent traversal, child traversal, child-exists semijoins, intersection, union, temp materialization, owner projection, relation optimization, count estimates, and exact-safe counts. Production stored and nested lowered execution must consume the opaque relation API: lower a relation inside the active read view, apply the requested domain through `relation_filter_domain()`, let the backend choose inline versus temp strategy through terminal-aware `relation_optimize()`, and fetch CDef pages through `iter_relation_cdef_batches()`. Query-backed page terminals should avoid read-view-local full temp materialization unless a backend can keep the relation session across pages. Backend-private plan objects may exist behind that API, but orchestration code must not call plan paging directly. A `DefinitionResultSet` or cursor must not retain a live SQLite cursor, connection, transaction, or backend-local ID without source/generation metadata.

Lowered graph plans are anchor-oriented. A backend should report the chosen anchor node, anchor reason, anchor relation kind, anchor estimate, logical relation summary, physical relation strategy, inline/materialized relation names, parent/child propagation steps, page fetch counts, terminal stop reason, scan policy, verify budget, and count witness/collision diagnostics when relevant. SQLite uses `definitions.stable_hash` for stable-hash anchors and `postings.feature_id` for posting anchors, walks `definition_edges` in SQL to project the selector root, applies domain filters in SQL, and leaves final truth to Python `ConcreteDefinition` verification. SQLite `EXPLAIN QUERY PLAN` rows are diagnostics requested with `explain(sql=True)`, not a public semantic contract.

Exact stored-root `ConcreteDefinition` count, exists, and definition-returning terminals are exact-safe after backend equality confirmation and may bypass final query verification. Stable-hash candidate anchors for general selectors are not exact-safe; Python verification remains authoritative for those paths.

## Transaction Boundaries

SQLite writes use one logical transaction per mutation:

- register graph nodes
- register local feature postings
- register direct edges
- activate or remove stored roots
- increment generation once if the logical state changed

Expensive pure work, such as graph traversal and row encoding, happens before acquiring the SQLite writer slot. The writer transaction is intentionally short and retried only for known busy/locked errors.

Read views begin a SQLite read transaction, capture the current DRYML generation, fetch candidate IDs/CDefs, and close before Python verification or user iteration. Result sets must not retain a SQLite cursor or transaction.

## Concurrency

SQLite connections are process/thread local. A forked or spawned worker opens its own connection. A coordinator with an existing connection sees committed worker writes on its next read transaction without reconnecting.

WAL mode can support readers while a writer commits when explicitly selected and available. The default `auto` policy is conservative and uses rollback journal unless the runtime is known safe for WAL.

## Recovery

Object files remain authoritative. A missing, dirty, corrupt, incompatible, stale, or divergent SQLite sidecar is rebuilt from Store roots. Corrupt sidecars are quarantined before rebuild. A misplaced or changed `def.pkl` is Store corruption and is reported instead of silently reindexing under the wrong hash.

## Helper Discipline

Shared persistent semantics live in focused modules:

- `query/codecs.py` owns CDef, feature-token, and graph-path encoding.
- `query/utils.py` owns shared identity/hash utilities.
- `query/sqlite/utils.py` owns SQLite WAL-safety and busy-error classification.
- `query/sqlite/connection.py` owns connection lifecycle.
- `query/sqlite/index.py` owns SQLite query-index operations.

Planner and query orchestration code must not import SQLite implementation modules.

The V2 helper audit kept terminal policy in `query/lowering.py`, SQLite SQL strategy in `query/sqlite/lowering.py`, and backend lifecycle in `query/sqlite/index.py`. No catch-all helper module was added.
