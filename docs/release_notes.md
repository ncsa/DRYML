# Release Notes

Status: draft.

## 0.3.0.dev0 (unreleased)

This development release consolidates the completed requirements, runtime,
dispatch, world, and code-analysis work:

- `dryml.session` is the documented common runtime path. Omitted session setup
  now intentionally means unchecked `RuntimeMode.NONE` Python rather than strict
  orchestrator enforcement; serialized `none` is explicit, and `manage()` and
  `set_mode("orchestrator")` are explicit opt-ins;
- session snapshots are immutable, separate the current-process allowance from a
  requested worker world, report per-control status, and retain safe
  reset/restart boundaries around framework imports and active generation leases;
- `session.request_world(...)` is removed; use
  `session.worker_world_request(...)` to set the default world for later
  dispatched workers;
- raw registered TensorFlow, PyTorch, and JAX root imports traverse lightweight
  hooks in managed/orchestrator sessions. Mandatory visibility fails closed;
  optional framework and allocator controls are reported per adapter without
  claiming aggregate process-memory or accelerator-memory enforcement;
- hard annotations are checked for supported direct calls in managed sessions,
  while Python mode remains unchecked. Managed sessions and Store-backed managed
  operations are distinct APIs;

- the active core implementation is promoted to its permanent `dryml.core`
  package; the temporary pre-release package route is removed without an alias,
  and persisted definitions, Stores, records, pickles, hashes, or import
  references that depend on that removed module identity are unsupported;
- reusable `dryml.code` facts/analyzers, conservative static calls, probes, and
  explicit current-process tracing;
- non-invoking `analyze(...)` versus trusted invocation-bearing `trace(...)`;
- the method semantic model at `dryml.core.methods`, with warning-free
  `dryml.code` compatibility aliases;
- class, method, function, and CDef requirement collection with source traces;
- runtime enforcement and inline `runtime.plain()`;
- context-local environment/world defaults, environment registry resolution, and
  lightweight local-world synthesis;
- Python-shaped function and CDef-method dispatch, requirement-aware candidate
  checks, strict/all-axis defaults, complete v2 worker-session bootstrap,
  `dispatch.explain(...)`, local subprocess/local-world boundaries, and explicit
  opt-in bounded dispatch tracing;
- one managed `compute`/`train` lifecycle with fenced DirStore writers, durable
  checkpoints, active/history selection, stale-input detection, exact lineage,
  explicit transfer, and protected cleanup;
- lightweight sharded CachedDataset realizations, NumPy/Parquet conversion, lazy
  TensorFlow/Torch views, and managed TensorFlow/Torch/sklearn training
  capability contracts;
- record-backed `CategoricalAccuracy` and `ConfusionMatrix` Artifacts with stable
  labels, true-row/predicted-column orientation, lightweight reload, and no
  implicit dependency computation;
- retirement of the unsupported tracked `dryml.graph` prototype package; and
- the [migration guide](migration/legacy_context_execute_removal.md), three
  [focused runnable scripts](table_of_content.md#focused-runnable-workflows),
  and the ordered [six-notebook tutorial set](table_of_content.md#tutorials).

The source distribution includes exactly those six authored tutorial notebooks
alongside the three focused scripts. Wheels continue to exclude `examples/`,
`docs/`, and `tests/`.

Strict orchestrator mode is intentionally breaking for local materialization,
local managed calls, and current-process trace execution: it is session-wide
definition-only control-plane mode with an accelerator-hidden `NoAllocation`
parent. Use definitions for planning and dispatch for workload execution. Future
worker environment/world requests remain candidates, while `require_env(...)`
remains hard compatibility. The boundary is a trusted-code lifecycle boundary,
not a sandbox.

Execution envelopes are now v2 and contain complete canonical environment, world,
runtime, allocation, policy, and axes selections. V1 execution envelopes are
rejected with migration guidance rather than upgraded from missing fields.

The accepted Sprint 11 release candidate completed the maintained local suite in
428.900 seconds with 2,523 smoke/medium passes, 143 heavy passes, and one
expected xfail. Its fixed coverage reference is 26,739 of 32,377 lines (82.59%).
Across supported Linux Python 3.10 through 3.14 jobs, the median test-step time
was 218.279 seconds; the supported Windows Python 3.12 test step completed in
496.344 seconds. These are accepted candidate measurements, not new performance
budgets or a claim of package publication.

Known limitations: source-backed subprocess reconstruction, cross-environment or
subprocess trace, trace sandboxing/hard timeouts, package solving, remote or
multi-host worlds, and cross-Python pickle transport are not implemented.
`allow_pickle=True` remains same-Python-only. Alias-aware static analysis, alias
provenance, and general Python call tracing remain deferred under
[ADR 0008](adr/0008-deferred-alias-aware-code-analysis.md).
An omitted dispatch `requirement_policy` now uses strict compatibility on all
axes in every caller mode. Explicit advanced policy/axis overrides relax
compatibility only: malformed operations, unsafe transport, allocation
feasibility, mandatory visibility, and worker protocol validation remain
blocking.

Real framework and GPU runtime evidence remains opt-in. The default suite uses
deterministic fake frameworks and does not claim that local installed packages or
an unrun GPU host prove TensorFlow, PyTorch, JAX, allocator, or GPU behavior.

The draft `dryml.artifacts.Accuracy` API and object-directory Artifact payload
contract are removed without compatibility aliases or migration. Use
`dryml.metrics.CategoricalAccuracy` with logical trained-model and completed
cache output refs. `Scalar` remains immediate definition data; Store-backed
`ScalarAgg`/`ScalarAvg.compute` results now use managed `DataRecord` products.
See [ADR 0009](adr/0009-managed-operation-lifecycle.md) for stable identity,
authority, recovery, Store, format, and export contracts.

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
