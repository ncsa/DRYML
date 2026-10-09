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
| lazy result iteration | `collect()` before ordinary iteration; use `take(n)` for a bounded prefix |
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
boundaries, exact save/load, and derived-sidecar recovery. The focused U1-U6 test
modules retain the complete error and parameter matrices; the representative profile
uses acceptance cases rather than expanding those products.
