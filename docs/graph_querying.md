# Graph Querying

Query V3 has one composable identity universe. `Repo.query()` captures identity
knowledge from connected Stores and retained caches; `Store.query()` captures one
Store's authority. Construction is inert. Restrictions and terminals inspect
definitions, complete ObjectRefs, StateRefs, and captured metadata without
materializing Objects, reading payloads, allocating ObjectIds, saving, observing
the host, or running workloads.

`Repo.query(weak=True)` includes both weakly and strongly retained cache facts;
pass `weak=False` when only strong cache retention may contribute. This producer
choice does not affect connected Store knowledge. `Repo.get()` and item lookup
remain non-query live-Object operations and return `ObjectResultSet` where their
existing API returns multiple retained Objects.

```python
from dryml.core import Definition

models = repo.query().sel(Definition(Model)).cdefs().stored().collect()
model = repo.load_object(models.one())
```

`sel()` composes structural Definitions, complete CDefs, `Selector` values,
Generator selectors, and exact references. Complete Definitions remain structural;
complete CDefs and exact references retain their complete graph identity. Query
results are `IdentitySet` values with deterministic identity ordering. `collect()`
fixes membership before iteration, while `take(n)` is the explicit bounded prefix
operation and retains visible boundedness on its returned set.

Exact live StateRef selection restricts the complete knowledge universe: embedded
references in definitions, materializing subtree projections, and retained cache
receipts participate alongside independent snapshots. `collect()` completes the
contributing-source evidence rather than stopping at a direct snapshot hit.
Knowledge does not imply independent `.stored()` membership or metadata-holder
eligibility. An embedded or projected StateRef without its own snapshot remains
known but is not independently stored.

Only an exact-StateRef `exists()` without alias, stored, metadata, cache-scope, or
source-evidence restrictions may use a validated direct positive without broad
inventory. This direct authority proof cannot satisfy `require_indexed()`. A
direct miss says nothing about embedded knowledge: a negative live
answer requires complete relevant inventory, or a visible refusal under a no-scan
policy. Other exact terminals, including analyzed explanations, complete that
inventory. Scoped scalar terminals validate their selected authority and captured
read failures before stopping at a positive candidate.

`ObjectRef.object_projection()` and `StateRef.object_projection()` produce an
`ObjectSelector` for one complete realized graph. Projection preserves ObjectIds,
CDef topology, and edge roles while replacing every encountered exact reference
with a nested `ObjectSelector`; StateRef hashes do not participate. Raw ObjectRef
or StateRef values that a caller explicitly embeds in an ObjectSelector remain
exact pins. ObjectSelectors are query-only, but their `reference` property exposes
the corresponding recursively state-weakened ObjectRef when a persistence API
requires object-association authority.

Use `query.object_projection()` or `identity_set.object_projection()` to collect
and deduplicate reference members as a fixed `ObjectSelectorSet`. CDef members are
omitted because they have no realized ObjectIds. This makes multiple checkpoints
of one realized composite graph count once without merging separately constructed
graphs:

```python
runs = repo.query().state_refs().object_projection()
run_count = runs.count()
```

`categorical()`, `exact()`, and `restore()` edit the latest selector without
discarding prior query restrictions. During categorical projection,
named drops are validated against the original
selected traversal before projection; a projected name cannot hide an invalid or
absent source path.

For a stored-CDef-only query whose Stores have ready SQLite indexes, `take(n)`
uses graph-complete `(graph digest, encoded root)` keyset pages and a bounded
cross-Store frontier. The encoded tie key preserves sharing topology and genuine
digest collisions. Query V3 verifies returned CDefs against direct Store authority,
detects unmanaged stored-root sidecar mutation, and otherwise falls back to complete
authority unless `scan_policy("forbid")` or `require_indexed()` requires a visible
failure. `refresh(False)` uses this path only when the sidecar is already ready; it
never initializes or rebuilds derived state.

## Identity Restrictions

Use `cdefs()`, `object_refs()`, and `state_refs()` as lazy kind restrictions.
`stored(scope=...)` restricts the existing universe to kind-specific authoritative
membership; it never discovers extra identities. `cached(scope=..., weak=...)`
similarly restricts an existing universe to retained cache knowledge. `in_source()`
limits contribution evidence and binds a later default authority source; use an
explicit `scope=` for metadata, aliases, or membership authority when a fixed or
multi-producer expression has no unambiguous default.

Reference identity restrictions compose in the same plan: `object_id()`,
`namespace()`, `contains()`, `state_hash()`, `alias()`, and `where(field(...))`.
Metadata scopes are `object`, `state`, `lineage`, and `snapshot`; terminals retain
authority validation and report conflicts or malformed records rather than treating
them as non-matches.

Within one terminal, complete captures share validated definition/root inventory,
derived knowledge, and independent membership facts. Reference validation uses
the complete selected authoritative-root combination before filtering, without
promoting embedded references into independent ObjectId authority. Successful
combination proofs are reused by sorted unique authority-fence domains and
terminal-local capture tokens. Replacement facts always have a fresh token,
including metadata-demand recapture with unchanged authority, and restart dependent
results. SQLite generations are not Store authority generations. Repo topology,
cache tiers, full knowledge, and retained root-or-cache lookup remain distinct;
separate Zip transactions retain separate cuts. Proofs never cross terminals.

## Relationships

Relationships begin with an explicitly selected root universe. `roots.nested()`
returns strict non-empty typed occurrences. Its default `EdgePolicy.ALL` includes
associations, materializing edges, and retained references. Pass `EdgePolicy.OWNED`
when owning-only traversal is intended. `through(RelationshipKind.REFERENCE)`
qualifies actual paths containing a reference hop. `owners()` and `targets()` return
new identity queries, so their result kind is selected with a normal kind
restriction.

```python
from dryml.core.query import EdgePolicy, RelationshipKind

owners = (
    repo.query().cdefs().stored()
    .nested(target, edges=EdgePolicy.ALL)
    .through(RelationshipKind.REFERENCE)
    .owners().cdefs().collect()
)
```

Raw occurrence terminals retain root, typed path, target, and contributing-source
evidence. `max_occurrences(n)` bounds only raw occurrence output; direct owner and
target projections remain existential. `max_depth()`, `max_verify()`,
`max_witnesses()`, `scan_policy()`, `require_indexed()`, and `explain()` preserve
visible execution bounds and diagnostics across composition. `explain(analyze=True)`
executes the plan; ordinary explanations report planned source cuts and residual
work without evaluation.

`scan_policy("allow")` permits required authoritative inventory scans,
`scan_policy("warn")` additionally emits one warning per scanning terminal, and
`scan_policy("forbid")` rejects the terminal before scanning. `refresh(False)`
does not reconcile derived indexes, `refresh("auto")` reconciles stale indexes
when safe, and `refresh(True)` forces reconciliation before each terminal cut.
Refresh may replace derived index sidecars but never rewrites Store authority.

These are terminal-owned controls: they apply to producers **and** external
stored, alias, and metadata scopes, algebra operands, closure, occurrence targets,
and owner/target projections. A fixed producer is source-free only when its
restrictions are source-free too. Projections inherit controls without inheriting
raw occurrence caps. An explicit control on the outer query governs its nested
execution; algebra construction still rejects incompatible operand policies.
Warnings precede the first inventory or recovery scan, once per terminal.
Forced refresh does not permit a recovery scan under forbid or indexed-only
controls; disabled refresh does not initialize or rebuild a missing/dirty sidecar.

### Verification Admission Budget

`max_verify(n)` bounds cumulative graph-exact detached identity candidates admitted
to verification by one terminal, rather than filesystem reads, SQL rows, bytes,
metadata holders, or replica checks. Budget exhaustion raises
`QueryVerifyBudgetExceeded` before the next candidate/domain is verified.

- Ordinary inventory reserves the **whole detached candidate domain** before
  restrictions or scalar early stopping. Fixed refinement reserves its fixed
  domain; reading an external authority scope does not admit all its holders.
- Exact checks and indexed pages reserve candidates before direct verification,
  including misses and rejected candidates. Identical replicas and repeated stages
  share one admission within an unchanged cut. Genuine collision identities are
  distinct admissions.
- Ordering-only, unverified lookahead counts fetch/decode work, not admissions.
  Lookahead verified to complete a collision bucket is charged even if not returned.
- Replacing a cut, including unchanged metadata-demand recapture or instability
  retry, discards reusable admissions and charges readmission. Spent terminal work
  is never refunded. Selective/paged fallback charges its new complete inventory
  domain in addition to abandoned-attempt expenditure. Existing retry caps remain.
- `take(0)` acquires no sources and admits nothing. Witness and depth budgets are
  separate and inherit effective outer controls.

Analyzed explanations report admissions as `verified_count`, inventory/completed
recovery scans as `store_scan_count`, and page work separately as
`candidate_rows_read`, `cdef_blobs_decoded`, and `pages_fetched`. Payload-free
metadata and recovery IO remain scan-controlled work, even without admissions.

Trusted selector resolution may import ordinary Python code when its existing
semantics require it. That accepted import side effect does not permit query
operators to construct Objects or execute workloads.
