# SQLite Query Lowering

DRYML's SQLite query index lowers safe selector work into SQLite while keeping Python `Definition` matching authoritative.

Lowered SQLite execution currently handles:

- backend-owned candidate relation contracts identified by source key, generation, opaque relation ID, relation kind, stable ordering, estimated rows, exact-safe status, and debug label;
- active read-view relation operations for domain filtering, parent traversal, child traversal, child-exists semijoins, intersection, union, temp materialization, owner projection, relation optimization, keyset paging, count estimates, and exact-safe counts;
- anchor-first relation plans for exact stable-hash and local-posting anchors;
- local feature posting predicates;
- parent edge projection from rare nested anchors and child edge projection for sibling subtrees;
- stored-root domain filtering through the read-view relation API on production lowered paths;
- nested-domain filtering through recursive ancestor traversal;
- owner projection through `relation_project_owners()` on a verified nested-target relation, with owner count diagnostics;
- stable keyset-ordered candidate batches using `(stable_hash, collision_ordinal, def_id)` ordering;
- terminal-aware `exists()`, `one()`, `one_or_none()`, and `count()` execution using terminal-sized keyset pages where possible, with exact stored-root definition terminals using exact-safe backend paths;
- plan-only `explain()`, opt-in `explain(sql=True)` SQLite plan diagnostics, and analyzed `explain(analyze=True)` diagnostics with logical-plan, physical-plan, scan-policy, and verify-budget fields;
- scan policies through `scan_policy("allow" | "warn" | "forbid")`, `require_indexed()`, and `max_verify(...)`.
- query-backed `DefinitionResultSet` paging for broad stored SQLite queries above the eager threshold.

SQLite lowering is conservative. SQL may return false positives, but returned definitions are still verified in Python with the normal query matcher. SQL must not introduce false negatives for supported lowered predicates.

Read transactions remain short. Candidate IDs and CDef batches are fetched inside a read view, the read view closes, and Python verification runs afterward. Production stored and nested lowered execution lowers an active `CandidateRelation`, composes the requested domain with `relation_filter_domain()`, and pages through `iter_relation_cdef_batches()`. The legacy plan pager remains a SQLite-private implementation detail behind the relation pager. Result metadata records candidate rows read, CDef blobs decoded, Python verifications, pages fetched, scan fallback reason, terminal stop reason, anchor node/reason/estimate, logical relation summary, physical relation strategy, inline/materialized relation names, propagation direction, SQLite plan rows when requested, count witness reloads, count collision buckets, and per-source generation.

SQLite applies a deterministic `SQLiteOptimizerPolicy` inside the active read view. The current policy keeps small single-use relations inline and materializes relations when they are reused, exceed the row-estimate threshold, exceed the SQL-length threshold, or feed recursive owner projection. The default thresholds are intentionally simple backend constants rather than public API: materialize if reused, if estimated rows exceed 10,000, or if SQL text exceeds 20,000 characters. Page-terminal relations stay inline by default because query-backed paging opens a fresh read view per page and read-view-local temp relations cannot be reused across page fetches. Physical diagnostics report the actual choice as `inline-cte` or `temp-relation` with the reason used for materialization.

For selector graphs with indexable requirements, SQLite lowering chooses the rarest exact stable-hash or local-posting node as the SQL anchor. Exact anchors start from `definitions.stable_hash`; local-posting anchors start from `postings.feature_id` and `postings.multiplicity`, then apply remaining local predicates. If the anchor is nested, SQL builds that relation first, walks `definition_edges` toward the selector root with path predicates, applies local predicates at each reached parent, applies sibling child-subtree checks with `EXISTS`, then projects the root relation and applies the requested domain filter. Exact anchors are stable-hash candidate anchors; Python verification remains authoritative for hash-collision buckets and all selector semantics.

Lowering currently chooses one anchor path to the selector root. Sibling subtrees are enforced as SQL `EXISTS` filters and relation operations expose SQL-native intersection/union/materialization for backend-owned set composition. A full multi-anchor cost-based optimizer is future scale work.

Query-backed `DefinitionResultSet` ordering is stable in source order, with each source ordered by `(stable_hash, collision_ordinal, def_id)`. This source-order contract is the current optimizer-sprint cross-Store paging policy; global keyset merge across Stores is deferred. Each page rebuilds the generation-bound relation inside a fresh read view and fetches through `iter_relation_cdef_batches()` using an opaque `PagedResultCursor` containing source key, generation, and the last ordering key. Because each read view is short-lived, the page terminal does not full-materialize broad relations into read-view-local temp tables by default. After materialization, repeated iteration preserves the original streamed page-factory order rather than re-sorting cached results. A result set stores a generation vector and opaque keyset page cursors, but it does not retain SQLite connections, cursors, or live read transactions. If a source generation changes before page iteration completes, iteration raises `QueryIndexGenerationChanged` instead of silently mixing snapshots.

Lowered `count()` streams verified CDefs into a collision-safe stable-hash count state. On first sight of a stable hash it stores only `(source_key, generation, def_id)` as a witness ref and increments the count. If the same stable hash appears again, federation reopens a short backend read view to load that witness CDef and materializes a collision bucket for that hash only. If the witness source generation changed before reload, the count terminal retries from the beginning a bounded number of times. Non-colliding result sets therefore do not retain full CDefs after verification, while duplicate definitions and stable-hash collisions remain exact. Exact stored-root `ConcreteDefinition` count, exists, and definition-returning terminals use exact-safe relation paths after backend equality confirmation and do not run final query verification or fetch candidate CDef pages.

Fallback boundaries:

- memory indexes use the v1 set-based path;
- SQLite falls back to broad candidate relations for unindexed, graph-shaped no-indexable, or callable-only selectors unless scan policy forbids it;
- small definition results still use eager `DefinitionResultSet` snapshots;
- broad query-backed definition results store a generation vector and fail clearly if a Store generation changes before page iteration;
- arbitrary Python callable selector semantics are never evaluated in SQLite;
- object materialization remains explicit through `objects()` after definition search;
- occurrence path enumeration remains Python-side and lazy, fed by verified lowered nested target IDs. `max_occurrences` bounds emitted paths; current backing capture is candidate-target scoped and reports nested targets, captured nodes, captured incoming edges, owners found, and the configured path limit. It is not a full SQL-native path enumerator, and lazy result-set explanations do not know the eventual emitted path count until iteration. Full SQL-native occurrence path enumeration is deferred.

The million-definition benchmark contract is tracked in `docs/sqlite_lowering_million_definition_benchmark.md`.

The helper policy for this lowering work is to keep SQL strategy code in `query/sqlite/lowering.py`, backend-independent terminal state in `query/lowering.py`, and shared persistent codecs/utilities in their existing focused modules. No catch-all helper module should be added.
