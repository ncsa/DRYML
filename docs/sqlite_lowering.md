# SQLite Query Lowering

DRYML's SQLite query index lowers safe selector work into SQLite while keeping Python `Definition` matching authoritative.

Lowered SQLite execution currently handles:

- backend-owned candidate relation contracts identified by source key and generation;
- exact stable-hash candidate anchors;
- local feature posting predicates;
- direct child edge joins for selector subdefinitions;
- stored-root domain filtering;
- nested-domain filtering through recursive ancestor traversal;
- owner projection through the existing recursive SQLite owner relation;
- stable keyset-ordered candidate batches using `(stable_hash, collision_ordinal, def_id)` ordering;
- terminal-aware `exists()`, `one()`, `one_or_none()`, and `count()` verification that stops within fetched pages;
- plan-only `explain()`, optional `explain(sql=True)` SQLite plan diagnostics, and analyzed `explain(analyze=True)` diagnostics;
- scan policies through `scan_policy("allow" | "warn" | "forbid")`, `require_indexed()`, and `max_verify(...)`.
- query-backed `DefinitionResultSet` paging for broad stored SQLite queries above the eager threshold.

SQLite lowering is conservative. SQL may return false positives, but returned definitions are still verified in Python with the normal query matcher. SQL must not introduce false negatives for supported lowered predicates.

Read transactions remain short. Candidate IDs and CDef batches are fetched inside a read view, the read view closes, and Python verification runs afterward. Result metadata records candidate rows read, CDef blobs decoded, Python verifications, scan fallback reason, terminal stop reason, SQLite plan rows when requested, and per-source generation.

Query-backed `DefinitionResultSet` ordering is stable in source order, with each source ordered by `(stable_hash, collision_ordinal, def_id)`. A result set stores a generation vector and opaque keyset page cursors, but it does not retain SQLite connections, cursors, or live read transactions. If a source generation changes before page iteration completes, iteration raises `QueryIndexGenerationChanged` instead of silently mixing snapshots.

Fallback boundaries:

- memory indexes use the v1 set-based path;
- SQLite falls back to broad candidate relations for unindexed, graph-shaped no-indexable, or callable-only selectors unless scan policy forbids it;
- small definition results still use eager `DefinitionResultSet` snapshots;
- broad query-backed definition results store a generation vector and fail clearly if a Store generation changes before page iteration;
- arbitrary Python callable selector semantics are never evaluated in SQLite;
- object materialization remains explicit through `objects()` after definition search;
- occurrence path enumeration remains Python-side and lazy, fed by verified lowered nested target IDs. Full SQL-native occurrence path enumeration is deferred.

The million-definition benchmark contract is tracked in `docs/sqlite_lowering_million_definition_benchmark.md`.

The helper policy for this lowering work is to keep SQL strategy code in `query/sqlite/lowering.py`, backend-independent terminal state in `query/lowering.py`, and shared persistent codecs/utilities in their existing focused modules. No catch-all helper module should be added.
