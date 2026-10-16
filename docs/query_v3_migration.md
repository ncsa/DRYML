# Query V3 Migration

Query V3 replaces separate structural and reference builders with a single
identity-universe API. Start with `repo.query()` or `store.query()`, compose
restrictions, and finish with an explicit terminal. This guide intentionally names
retired spellings only to map existing applications to their replacement.

## Entry Points And Results

| Previous area | V3 expression |
| --- | --- |
| `Repo.query(selector)` | `repo.query().sel(selector)` |
| `Repo.references()` | `repo.query()` plus `object_refs()` or `state_refs()` |
| definition/reference result families | fixed `IdentitySet`; relationship paths use `OccurrenceSet` |
| stored, cached, known domain selection | producer knowledge by default; add `stored()` or `cached()` only to narrow it |
| strong-only versus weak Repo cache producer | `repo.query(weak=False)` for strong-only initial knowledge; the default includes both tiers |
| `find_defs`, `find_occurrences`, `find_owner_defs` | compose an identity or occurrence query and call `collect()` |
| `find`, `find_owners` | select identities, then call an explicit Repo load method |
| lazy result iteration | explicit `.iter()` for incremental consumption; `collect()` for a fixed iterable result; `take(n)` for an eager bounded prefix |
| `defs()`, `execute()`, `refine()` | `collect()`, scalar terminals, or `result.query().sel(...)` |

`Repo.query()` and `Store.query()` provide their producer's initial universe, so
normal terminals need no preliminary domain call. `IdentitySet.query()` is retained
and can only refine its captured members. `union()` and `intersection()` accept
queries from different producers, preserve source evidence, and deduplicate complete
typed identities. Result ordering is deterministic by complete identity; it does not
mean newest, best, or completed.

`object_projection()` is an explicit terminal on `IdentityQuery` and a source-free
projection on `IdentitySet`. It maps ObjectRef and StateRef members to recursive
`ObjectSelector` values, omits CDefs, merges evidence for equal projections, and
returns a fixed `ObjectSelectorSet`. Exact reference selection remains unchanged;
an ObjectSelector is the separate state-independent graph-selection value.

`.iter()` now returns a single-use closeable generator on both identity and
occurrence queries. It starts on first advancement and matches/traverses on
demand, retaining normal upfront candidate, authority and metadata validation.
Close early-abandoned iterators with `contextlib.closing`; a `break` alone does
not close a retained generator. See [Lazy Iteration](graph_querying.md#lazy-iteration)
for snapshot, ordering, ownership, error and memory semantics.

The producer option `repo.query(weak=...)` controls which cache tiers enter the
initial universe. It is distinct from `query.cached(scope=..., weak=...)`, which
only narrows identities already present in that universe.

## Selection, Membership, And References

`sel()` retains structural Definition matching, class policies, strict matching,
partial fields, exact CDef topology, Ref/Mat roles, and complete reference leaves.
Use `categorical()`, `exact()`, and `restore()` after a selector restriction; each
adds immutable intent without resetting earlier restrictions. Loose symbolic
selection and exact Generator support remain distinct: exact support still requires
proof rather than accepting a prefilter as an answer.

Live Objects are not universe members. Use their retained definition for structural
intent, or an exact ObjectRef/StateRef for identity intent. `cdefs()`,
`object_refs()`, and `state_refs()` are restrictions, not projections that load or
construct data. ObjectId, namespace, exact-reference, alias, subtree, and state-hash
searches compose directly on the identity plan. `stored(scope)` narrows existing
members by kind-specific authoritative membership and never fetches candidates.

The retired reference `objects()` and `states()` aliases become a kind restriction
followed by a terminal. The retired definition/result `objects()` materialization
becomes explicit `repo.load_object(cdef)` or another explicit Repo load API after
identity selection. `Repo.get`, item access, and explicit load methods remain
non-query APIs with their established candidate and error behavior.

## Metadata, Traversal, And Scope

`where(field(...))` remains a restriction in the same identity universe and composes
with selection, kinds, membership, and algebra. Its `object`, `state`, `lineage`,
and `snapshot` fields retain their existing operators, validation, current-versus-
captured evidence meanings, and conflict errors. A state-metadata match can be
expanded with relationship operations and then restricted to owning ObjectRefs when
that association is intended; it does not silently become a structural object match.

The retired containment adapter becomes an explicit root universe followed by
`roots.nested(selector, edges=...)`. Nested paths are always non-empty. The default
`EdgePolicy.ALL` includes all retained relationships; use `EdgePolicy.OWNED` for
owning-only behavior. Occurrences retain typed paths and hop evidence; `owners()` and
`targets()` produce composable identity queries. Replace `contains_ref=True` with
`through(RelationshipKind.REFERENCE)` on the occurrence plan.

Replace `in_store()` with producer selection, `in_source()` for contribution scope,
and explicit `scope=` for metadata, aliases, or membership authority. Source
evidence identifies contributors but is not an implicit loading hint; inspect
`sources(member)` and choose an explicit Store when selecting a persisted copy.

## Bounds, Diagnostics, And Effects

Use `take(1).one_or_none()` instead of implicit paged first access. `take`,
`max_occurrences`, `max_depth`, `max_verify`, and `max_witnesses` keep boundedness
visible; a bound never silently claims complete output. `count`, `exists`, `one`,
and `one_or_none` apply to both identity and occurrence result domains. `explain()`
describes stages, sources, result domain, residual work, and bounds; analyzed or
backend detail is opt-in. Refresh and scan/index policies remain explicit controls.
`IdentitySet.diagnostic()` returns the exported `QueryDiagnostic` type. Member and
source counts are disclosure-capped, while `candidate_rows_read`,
`cdef_blobs_decoded`, and `pages_fetched` retain exact keyset-page work through
fixed-result refinement and algebra.

## Refresh Lifecycle Migration

| Previous intent | Current API |
| --- | --- |
| Refresh once, then reuse branches with fresh authority | Keep one `query.refresh(True)` universe; ordinary derivatives share its request |
| Repeat explicit maintenance at every terminal | `query.refresh("always")` |
| Request another cycle after completion | Call `query.refresh(True)` again |
| Replace a persistent index unconditionally | `store.rebuild_query_index()` |
| Read current authority without maintaining derived state | `query.refresh(False)` |
| Recover only indexes consumed by execution | `query.refresh("auto")` (the default) |

Controls now survive owner/target projections and govern external stored, alias,
and metadata scopes as well as nested producers. Algebra operands must agree on
policies at composition; a later explicit outer override applies throughout the
expression. `refresh(False)` does not initialize/rebuild missing or dirty sidecars,
and explicit `refresh(True)` / `refresh("always")` cannot authorize prohibited
scans, including ready-index validation or joining an in-flight cycle. Query
`refresh(True)` is now **one-shot**, shared by ordinary derivatives and shallow
copies. A second explicit `refresh(True)` creates a new request. Use
`refresh("always")` to retain repeated per-terminal cadence. Ready cycles validate
without forcing replacement, so `"always"` does not reproduce the old unconditional
rebuild behavior; use `rebuild_query_index()` when unconditional replacement is
required. Direct backend calls remain independently repeatable one-cycle operations.
`Repo.refresh_index(force=True)` likewise requests reconciliation on every call;
its `force` parameter does not force replacement of a correct ready SQLite sidecar.

```python
once = repo.query().refresh(True)
once.cdefs().count()              # explicit cycle, then fresh evaluation
once.state_refs().collect()       # fresh authority; completed targets use auto
once.refresh(True).count()        # new one-shot request

repeated = repo.query().refresh("always")
repeated.count()                  # validates a correct ready sidecar
repeated.count()                  # validates again; does not force replacement
store.rebuild_query_index()       # explicit replacement when required
```

Combining completed disjoint universes preserves operand-local bindings and adds no
cross-operand cycles. New external scopes extend inherited requests, coalescing
pending obligations at a target when no older attempt is in flight. An outer True
creates one new request over the whole expression; False/auto/always suppress
inherited one-shot intent on that expression without changing originals. Closure,
occurrences, and both projections share lineage; refinements of fixed IdentitySet
and OccurrenceSet results are detached. Empty results retain source/scope obligations,
empty available Stores participate, and optional absent targets remain pending.

Same-request callers single-flight each target and observe exact joined outcomes;
later-created requests receive no historical credit. Failed/interrupted targets
remain retryable in later terminals, while successful targets survive partial or
evaluation failure. Owners retain original exceptions; failed/interrupted-owner
waiters receive `QueryIndexError`. Waiter interruption leaves the owner running,
and same-thread self-reentry raises `QueryIndexBusy`. Completed requests use fresh
terminal authority and auto demand behavior, including otherwise-supported indexed
prefixes under forbid/indexed-only controls.

Auto refresh is now demand-driven. Broad authority and metadata queries no longer
clean unused dirty sidecars as an incidental side effect. Use an explicit cycle
when maintenance itself is required; actual bounded index consumers still recover
outside retained readers/fences. Authority visibility remains fresh under every
policy. Ordinary explanation, construction, and identity/occurrence `take(0)`
remain inert. Optional absent indexes do not become fabricated targets; explicitly
configured unavailability and explicit maintenance failures surface to the caller.

Configure one persistent destination consistently per Store dirty-token domain,
including across clients. Terminal-local checks reject visible conflicting
destinations, ownership, or settings with `QueryIndexIncompatible`; independently
maintained destinations sharing that domain across clients remain unsupported.
Repo topology is retained once per terminal, so later attachments participate next
time. After fork, open a fresh Repo: inherited topology raises `RuntimeError`
before its owner lock. Shared request state resets to pending before child
acquisition, subject to existing Store/import/SQLite lifecycle rules. Requests
remain process-local, with no new query serialization or deepcopy contract.

See [Graph Querying](graph_querying.md#refresh-lifecycle-and-scan-controls) for the
full lineage, cohort, interruption, and process-local lifecycle contract, and
[Backend Contracts](query_index_backend_contracts.md#explicit-backend-reconciliation)
for strict maintenance success versus report-oriented reconciliation.

## Verification Budgets And Maintenance Diagnostics

`max_verify` has a tightened **cumulative admission** contract. Ordinary inventory
and fixed refinement reserve the entire detached domain before filtering or
early-stop terminals; exact/paged paths reserve before direct verification.
Repeated stages and replicas share unchanged-cut admissions, but collision
identities are distinct. Cut replacement (including unchanged metadata-demand
recapture), instability retry, and selective/paging fallback retain spent work and
charge new admissions. A budget formerly sufficient for each stage separately may
now fail with `QueryVerifyBudgetExceeded`. This accounting tightening is separate
from fixing the omitted budget check on indexed `take()`. Unverified ordering
lookahead counts rows/decodes only; verified collision lookahead is charged.
Metadata-holder discovery and recovery reads are scan-controlled IO rather than
identity admissions. `take(0)` reads/admits nothing; witness/depth budgets remain
separate and inherit controls. Analyzed explanations expose admissions as
`verified_count` and started inventory/maintenance scans as `store_scan_count`,
including failed recovery, fallback, and recapture. `refresh_action` reports actual
`rebuild > validate > waited > reused > none` by precedence across targets;
waiters receive no owner scan charges and reuse is not new reconciliation.
The declared `refresh` policy is reported separately.

Query authority is Store records, not memory or SQLite indexes. Derived indexes may
accelerate work, but missing, stale, corrupt, or incompatible indexes rebuild, fall
back visibly, or fail without changing authoritative records. Query V3 does not read
payloads, construct Objects, allocate ObjectIds, save, observe host state, or execute
workloads. Trusted ordinary selector resolution may import Python code when required
by existing selection semantics; that accepted import side effect does not widen the
query operator effect boundary.

## Acceptance Coverage

`tests/core/test_query_v3_acceptance.py` is the compact cross-layer acceptance
matrix. Its named cases cover stored authority and metadata restrictions, typed
reference traversal, fixed-set algebra and bounded terminals, query-effect
boundaries, exact save/load, and derived-sidecar recovery. The owning focused test
modules retain the complete error and parameter matrices. The representative profile
also selects named lifecycle regressions for one-shot/repeated cadence, operand-local
lineage, scope extension, controls, fresh authority, and publication failure.
SQLite backend/concurrency tests retain reconciliation, single-flight, joined
outcomes, interruption, fork/reset, ownership, and empty-token cleanup coverage
without curated function lists. Explicit exhaustive-only products remain outside
routine selection; sampled products retain representative values on every axis.
