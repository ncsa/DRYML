# Release Notes

Status: draft.

## 0.3.0-dev (unreleased)

This development release consolidates the completed requirements, runtime,
dispatch, world, and code-analysis work:

- reusable `dryml.code` facts/analyzers, conservative static calls, probes, and
  explicit current-process tracing;
- non-invoking `analyze(...)` versus trusted invocation-bearing `trace(...)`;
- the method semantic model at `dryml.core2.methods`, with warning-free
  `dryml.code` compatibility aliases;
- class, method, function, and CDef requirement collection with source traces;
- runtime enforcement and inline `runtime.plain()`;
- context-local environment/world defaults, environment registry resolution, and
  lightweight local-world synthesis;
- Python-shaped function and CDef-method dispatch, requirement-aware candidate
  checks, `dispatch.explain(...)`, local subprocess/local-world boundaries, and
  explicit opt-in bounded dispatch tracing;
- retirement of the unsupported tracked `dryml.graph` prototype package; and
- the [migration guide](migration/legacy_context_execute_removal.md) and four
  [runnable examples](table_of_content.md#focused-runnable-workflows).

Known limitations: source-backed subprocess reconstruction, cross-environment or
subprocess trace, trace sandboxing/hard timeouts, package solving, remote or
multi-host worlds, and cross-Python pickle transport are not implemented.
`allow_pickle=True` remains same-Python-only. Alias-aware static analysis, alias
provenance, and general Python call tracing remain deferred under
[ADR 0008](adr/0008-deferred-alias-aware-code-analysis.md).

## SQLite Query Index Sprint

The core repo/query path now supports Store-owned persistent SQLite query indexes for `DirStore`.

Highlights:

- `DirStore(query_index="auto" | "sqlite" | "memory" | "none")` controls query-index backend policy.
- SQLite indexes persist graph nodes, local feature postings, direct definition edges, stored-root membership, semantic versions, and generation metadata.
- Repo queries federate Store indexes and the live cache overlay while preserving Store-priority replica metadata.
- Existing ready SQLite indexes support stored and nested queries without hydrating every Store root.
- Saves publish object files before activating roots in the index; dirty markers and reconciliation recover object/index divergence.
- Multi-process workers use process-local SQLite connections and committed writes become visible to coordinator read transactions.
- Index status, validation, rebuild, and reconciliation are exposed through `Repo` and `DirStore` administration APIs.

Known deferred work:

- Public query-backed paginated ResultSets are not implemented yet.
- SQL-native candidate relations are deferred.
- Online side-by-side rebuild with manifest switching is deferred.
