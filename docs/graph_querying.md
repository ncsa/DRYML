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
digest collisions. Readiness and permitted recovery are resolved before any retained
reader. Unique Store authority fences are then acquired in sorted domain order;
non-repairing SQLite transactions recheck dirty tokens, schema/build readiness,
stored-root guards, and the actual sidecar UUID/generation/file identity. These
views and fences remain held through current main inspection, all pages, and direct
candidate verification. Main is checked even before an empty page: a missing or
malformed main target is an authority error, and a valid main outside direct-root
coverage requires complete fallback or explicit refusal.

Query V3 verifies returned CDefs against direct Store authority,
detects unmanaged stored-root sidecar mutation, and otherwise falls back to complete
authority unless `scan_policy("forbid")` or `require_indexed()` requires a visible
failure. `refresh(False)` uses this path only when the sidecar is already ready; it
never initializes or rebuilds derived state. Failed attempts close every view and
release every fence before fallback or recovery; frontiers and selective proofs
are discarded without refunding cumulative verification or IO work. Rebuild cannot
close another active same-process read view. Per-Store cuts do not promise a global
federated snapshot, broader V3 acceleration, or indexed exact-miss/count execution.

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

Complete inventory-backed bounded operands and selective fallback cuts participate
in this same combined validation. An explicit prefix cannot hide conflicts against
external authority scopes or another operand, including an empty filtered operand.
Selective positive and ready indexed-prefix paths still avoid broad inventory;
`take(0)` still acquires no sources.

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

Direct occurrence terminals and both projections combine complete root-producer
authority with external target metadata scopes before root or target filtering,
including empty domains. Metadata holders contribute authority proof only, not
candidate admissions or additional roots/targets.

`scan_policy("allow")` permits required authoritative inventory scans,
`scan_policy("warn")` additionally emits one warning per scanning terminal, and
`scan_policy("forbid")` rejects the terminal before scanning. `refresh(False)`
does no derived maintenance but still observes fresh authority. `refresh("auto")`
recovers only indexes actually consumed by the selected path. Broad authority
inventory, direct exact positives, and external metadata scopes leave unused dirty
sidecars untouched. A bounded stored-CDef index consumer preflights recovery outside
retained Store fences and SQLite readers, then enters non-repairing views.

`refresh(True)` creates a lazy **one-shot request**. Ordinary restrictions,
closure, raw occurrences, owner/target projections, and shallow copies share that
request; each applicable compatible destination completes it once, before a fresh
evaluation cut. A later terminal uses demand-driven auto behavior and obtains new
facts and proofs. Completion stores no inventory, result, reader, or authority proof.
Call `refresh(True)` again to create a fresh request; use `refresh("always")` for
one explicit cycle per applicable destination at every participating terminal.
An already correct ready sidecar is substantively validated without replacement;
repair may replace derived sidecars but never rewrites Store authority. Direct
backend `refresh(True)` and `refresh("always")` likewise perform one independent
cycle per call. Construction, ordinary `explain()`, and identity/occurrence
`take(0)` perform no maintenance or source acquisition. Explicit backend failures
propagate; absent optional backends create no maintenance target.

```python
universe = repo.query().refresh(True)
universe.cdefs().count()          # fulfills this request's current targets
universe.state_refs().collect()  # fresh evaluation, no repeat explicit cycle
universe.refresh(True).count()   # fresh request over the effective expression
```

Algebra preserves each operand's producer/scope bindings: combining independently
completed A and B universes does not refresh A for B's request or B for A's request.
A newly attached external authority scope C extends the expression's inherited
requests. In the absence of an older in-flight attempt, pending obligations for C
coalesce into one cycle. Empty filtered results retain scope obligations; even an
empty Store participates when its persistent target is available. Optional absent
targets receive no credit and participate when later available. New Repo attachments
introduce only their pending targets from the next terminal's retained topology.
An outer `refresh(True)` replaces inherited bindings with a new request covering
the whole expression. Outer `False`, `"auto"`, and `"always"` suppress inherited
one-shot obligations without changing original plans. Algebra still rejects
incompatible declared controls before an override. `IdentitySet.query()` and
`OccurrenceSet.query()` refine fixed results without inheriting live requests.

Same-request concurrent terminals single-flight each target. A terminal freezes its
applicable obligations, joins every encountered older attempt outside Store fences,
SQLite readers, and request locks, then rechecks pending pairs. It owns at most one
new cohort cycle per target and credits only pairs claimed by that cohort. Fresh
requests receive no historical credit. A joined failed attempt fails that terminal
even if a later retry already succeeded. Failed/interrupted targets remain retryable;
successful targets survive later target failure and evaluation/cardinality/budget
failure. Published attempt outcomes are immutable: interruption after publication
can fail the owner terminal without reversing completion or a joined outcome.
Owner exceptions propagate unchanged; interrupted-owner waiters receive
actionable `QueryIndexError`. Waiter interruption detaches without cancelling the
owner. Same-thread self-reentry raises `QueryIndexBusy` promptly. Independent
universes have no global single-flight guarantee.

Target resolution uses the terminal's first retained Repo topology, including all
producer and external-scope presentations before coalescing normalized configured
destinations. Authority domain, source key, dirty-token domain, canonical semantics,
backend settings, and backend/connection/configuration paths must agree; conflicts
raise `QueryIndexIncompatible` before maintenance. One consistently configured
custom destination per Store dirty-token domain is supported. Multiple visible
destinations sharing that domain are rejected, including inside one Repo. This is
a terminal-local check, not a cross-client or cross-process destination registry.
A Store attached after topology retention joins the next terminal. Inherited Repo
topology after fork raises `RuntimeError` before the parent-process configuration
lock; open a fresh Repo in the child. Query does not reset Repo-owned locks.
Request synchronization resets by PID before acquisition; inherited derivatives
share pending child request state. Direct Store execution remains subject to existing
Store, import, and SQLite lifecycle rules and uses fresh views. Requests are ephemeral
and process-local; there is no new serialization, transport, or deepcopy contract.

These are terminal-owned controls: they apply to producers **and** external
stored, alias, and metadata scopes, algebra operands, closure, occurrence targets,
and owner/target projections. A fixed producer is source-free only when its
restrictions are source-free too. Projections inherit controls without inheriting
raw occurrence caps. An explicit control on the outer query governs its nested
execution; algebra construction still rejects incompatible operand policies.
Warnings precede the first inventory or recovery scan, once per terminal.
Pending explicit cycles require scan admission even on ready sidecars, before
claiming or joining work; waiters cannot bypass forbid or indexed-only controls.
Completed requests can use otherwise-supported indexed prefixes under those
controls, with normal readiness and authority checks. Disabled refresh does not
initialize or rebuild a missing/dirty sidecar.

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

Analyzed explanations report admissions as `verified_count`, started inventories
and maintenance scans (including failed work, fallback, and recapture) as
`store_scan_count`, and page work separately as
`candidate_rows_read`, `cdef_blobs_decoded`, and `pages_fetched`. Payload-free
metadata and recovery IO remain scan-controlled work, even without admissions.
`refresh_action` reports actual actions rather than echoing the declared policy,
with precedence `rebuild > validate > waited > reused > none`. `waited` means
successful joining; `reused` means prior compatible request completion, not new
maintenance. Waiters never receive their owner's inventory charges.
Demand recovery reports its action too. Failed backend calls merge started work
during cleanup; retries never refund the terminal's cumulative ledger.

Trusted selector resolution may import ordinary Python code when its existing
semantics require it. That accepted import side effect does not permit query
operators to construct Objects or execute workloads.
