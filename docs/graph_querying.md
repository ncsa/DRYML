# Graph Querying

## Current Node Sources and Local-State Discovery

`Repo.query()` and `Store.query()` execute against Store 4 authority. Ready SQLite
schema-11/coverage-3 indexes supply contribution-scoped discovery postings. Rows reference
the existing authenticated root/block and portable table/overlay occurrence locator; they are derived
postings, not another authoritative graph representation.

Choose the question explicitly:

| Restriction | Domain |
|---|---|
| `cdefs()` | Represented recipe views; `stored()` requires independent recipe membership |
| `object_refs()` | State-weakened Node object projections |
| `saved()` / `state_refs()` | Complete saved Node graphs; `stored()` requires saved-root membership |
| `local_states()` | Distinct installed full-StateKey manifests, independent of donor ObjectIds |
| `local_states().available()` | Currently published data-directory availability, without payload authentication |
| `local_states().retained()` | Physical per-key retention grants, independently of managed lifecycle policy |

Source-backed graph results are Nodes. Domains remain tagged in collected
membership, including stateless graphs whose recipe and saved descriptions have
equal graph content. Node `==` compares immutable handles; use `graph_equal()` for
fresh-token semantic comparisons. A header or embedded dependency is not an
independently published root. Withdrawing one source contribution preserves other
replicas and retained immutable metadata.

An exact saved Node embedded through a Ref contributes saved **knowledge** even
when its containing publication is a recipe. It does not thereby acquire its own
saved-root membership or capture-metadata holder. Fixed collections and owner
projections preserve source domain tags, including stateless saved roots whose
graph content equals their recipe view. `contains(node)` accepts a current Node
object projection (and the retained ObjectRef adapter) and checks proper subtree
object-projection equality.

```python
states = (repo.query()
          .local_states(origin=creation_node)
          .available()
          .refresh(False)
          .require_indexed()
          .take(10))

exact = repo.query().local_states().state_key(known_key).refresh(False).one_or_none()
compatible = repo.query().local_states(compatible_with=destination_node).take(10)
saved = repo.query().saved().stored().sel(node_pattern).collect()
```

Exact full-key hit **and miss** use direct authority without opening SQLite or
enumerating buckets/graphs/cache knowledge. This proves only the installed-variant
domain. A hit reads and validates creation evidence separately. Exact-origin
selection uses the CDef bucket relation; compatibility fingerprints are necessary
prefilters followed by exact creation/configuration/codec checks. LocalState has
no donor identity: identity-constrained local-state predicates raise
`QueryDomainError`; select an explicit owner/occurrence instead.

Candidate predicates preserve typed paths, roles, and identity/state correlation.
Necessary predicates never replace exact Node/Generator residuals. All applicable
ObjectId anchors are checked in every participating Store, including Stores whose
candidate relation is empty. Missing/incompatible established anchors refuse the
query. Negative/count answers require the complete requested coverage and mutation
guards, independently of returned candidates.

ObjectId, full-StateKey and supplied local string/integer/boolean/None constraints
seek indexed local postings and direct occurrence-correlated edges, then join root
contributions. Local facts are stored once per block/view, not as each descendant's
complete feature list. Tied-set ordinals are presentation paths, never persisted
occurrence identity. Aliases, unordered assignment and quotations remain exact
residual obligations. A predicate without a supported safe anchor (for example,
class-only, negation-only or unsupported scalar/support-only selection) visibly
uses scan policy; `require_indexed()` and `scan_policy("forbid")` refuse before
refresh/recovery inventory. Explicit domain enumeration without a selector remains
available from its complete certified relation. Compatibility-only local-state
discovery currently needs scan admission; exact-origin discovery seeks its bucket.

`page_size(n)` bounds candidate pages. A short Store/SQLite cut captures candidate
keys and immutable authority bytes. Connections/transactions and writer-blocking
fences are released before shared block preparation, metadata expansion, residual
evaluation, and yielding. The page is accepted only after revalidating that same
cut. Continuation retains Store UUID/epoch/pending disposition, index file/UUID,
schema/coverage/generation, source topology, and a deterministic keyset cursor.
Alias, membership, availability, sidecar, coverage, or Repo topology changes cause
explicit `QueryError`/index instability, preserving an already-delivered partial
prefix; iteration does not silently restart. Federation uses per-Store cuts.

Composed stages sharing a terminal also reject a changed Store/index cut rather
than combining earlier and later evidence. Growing metadata demand on an
unchanged cut replaces detached facts and readmits candidates without refunding
work already charged to the terminal's verification budget. Requested metadata
scope validation precedes selector narrowing and scalar early stopping: a later
invalid populated field or conflicting holder cannot be hidden by a positive
match or an empty candidate set. Scope-holder evidence does not itself consume
candidate admissions or become part of the result universe.

`take(0)` is source-free. `refresh(False)` performs no maintenance but still checks
freshness. Missing/stale enabled coverage requires permitted recovery or visible
refusal. No-scan/indexed-only controls reject required inventory before it starts.
Ready explicit refresh validates compact certificates without historical inventory;
`rebuild_query_index()` requests metadata-only inventory. Disabled/memory indexing,
and automatic optional-backend unavailability, use visible metadata-only scan
fallback when permitted. Explicit SQLite requirements and foreign/future sidecars
refuse. Legacy standalone SQLite views do not cover Store-4 authority.

Pure decoded/prepared metadata is operation-local, conservatively byte-bounded,
and contains no open reader or lease. Fresh authority bytes and cut checks remain
mandatory on reuse. Pure caches never establish membership, names, availability,
retention, or completeness. Collected results fix membership; later payload use
requires its own fresh protected read.

## Node Pattern Semantics

Ordinary selection accepts a naked `Node` authored with `Definition(...)`.
The semantic compiler and detached metadata matcher are shared with the current
source consumer. Selection never concretizes omitted fields merely to obtain a
candidate key.

Only supplied constraints participate. Query preparation does not concretize,
fill defaults, construct an Object, allocate an ObjectId, capture state, or read
state data. A soft Node omits unconstrained constructor parameters. A
ConcreteNode anchors its complete concrete recipe and topology, without making
unspecified identity/state facets into absence constraints. Supplied ObjectId and
LocalState facets must match **the same corresponding occurrence**. LocalState
identity includes its full `(cdef_digest, codec_hash)` key and creation evidence,
never a donor ObjectId.

Shared Node handles require shared candidate nodes; independently authored equal
Nodes require independent nodes. Mat and Ref roles remain distinct; explicit Mat
wrappers and direct materializing children are equivalent. Quotation is exact
configuration, including its state/identity pins and omissions. Unordered graph
members use a whole-assignment witness, including joins outside the set.
`categorical(...)` can explicitly weaken recipe fields while retaining facets,
aliases and roles. Existing `Match` leaves remain supported; a runtime
`Satisfies` callable is not executed or serialized during graph preparation.
`Exact(...)` compares nested Node values semantically, including their alias
partition, roles, facets and quoted data; independent decoded handles need not
compare equal in Python. A fully specified `FactorySpec` matches its complete
argument/keyword shape. A factory containing `Match` leaves instead constrains
the supplied pattern fields and permits additional candidate keyword fields.

Use `QueryPredicate` only for the additional closed forms:

```text
predicate = Node
          | predicate & predicate | predicate | predicate | ~predicate
          | QueryPredicate.absent("identity" | "state", path=...)
          | QueryPredicate.support(generator, max_assignments=...)
```

```python
from dryml.core import Definition, QueryPredicate

pattern = Definition(Model, width=64)  # omitted fields stay unconstrained
predicate = QueryPredicate(pattern) & ~QueryPredicate.absent("state")
matcher = predicate.compile()         # reusable pure graph preparation
matches = matcher.matches(candidate_node)
```

`absent` requires an existing Node endpoint and constrains just that facet.
Omitting a facet is unconstrained; `AnyValue()` requires a present constructor
field; `Missing()` requires an absent constructor field. None is a wildcard for
exact missing state. Use `&`, `|`, `~`; Python truth testing raises TypeError.
Boolean evaluation does not hide unavailable exact support proofs in later
branches. Support applies a conservative prefilter followed by a correlated
assignment witness with the captured proof cap; bounds alone never accept.

`compile(domain="local_state")` evaluates descriptor metadata against its
creation evidence and exact state selection. It rejects donor-identity constraints
even in negated or otherwise masked branches. To constrain an actual donor,
select its relationship endpoint explicitly and match that Node with
`domain="occurrence"`. This endpoint mode does not discover relationships or
manufacture identity/state combinations. Generic `domain="node"` accepts recipe,
identified and saved graph metadata; it does not assert membership in any Store
domain. Source-domain counts, availability and retained/publication membership
remain separate source contracts.

`to_data()` / `from_data()` use the closed
`dryml-node-query-predicate` v1 family with one shared authenticated Node graph.
It preserves boolean forms, typed paths, aliases, fixed facets, Match leaves and
built-in `UniformIntRange` / `UniformFromSet` domains. Runtime Generator providers
are nonportable and rejected. Decoding never resolves targets or executes code.

Legacy `Selector`, `GeneratorSelector`, and reference selectors remain explicit
adapters. Legacy posting compilation retains an exact prepared residual and
visible reasons; unsupported facet/predicate coverage requires scan fallback or
refusal. Candidate IDs alone cannot certify Node acceptance or source completeness.

## Retained Fixed-Reference and Composition API

The following reference-specific descriptions apply to the retained fixed
ObjectRef/StateRef adapters. Store-4 source membership and page lifetimes are the
Node contracts above; Store-3 persistent inputs are not supported or migrated.

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
model = repo.build(models.one())
```

`cdefs()` returns concrete recipe Nodes, not necessarily complete saved graphs.
`repo.build(concrete_recipe_node)` constructs the selected recipe, allocating
missing identities and initializing unspecified state. To restore a complete
SavedNode selected with `saved()` / `state_refs()`, use `repo.load(saved_node)` or
`repo.load_state_ref(saved_node)`; loading does not initialize omitted state.

`sel()` composes structural Definitions, complete CDefs, `Selector` values,
Generator selectors, and exact references. Complete Definitions remain structural;
complete CDefs and exact references retain their complete graph identity. Query
collected results are `IdentitySet` values with deterministic identity ordering.
`collect()` fixes matching membership before iteration; `iter()` yields matches
on demand. `take(n)` is the explicit eager bounded-prefix operation and retains
visible boundedness on its returned set.

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

A selected Node is edited coherently at every alias of that occurrence. Restoring
one path in a shared graph therefore restores the same Node through its other
paths; independent equal nodes remain independent. Quoted constructor metadata
stays exact and separate from these graph edits. Non-Node value endpoints retain
ordinary structural-slot replacement semantics.

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

## Lazy Iteration

`IdentityQuery.iter()` is an explicit terminal returning a single-use, closeable
Python generator. Query plans themselves remain non-iterable. It yields the same
complete CDef, ObjectRef or StateRef values in the same identity order as
`collect()`, without constructing a final matching `IdentitySet`:

```python
for obj_ref in repo.query().sel(cached_test_ds_def).object_refs().iter():
    print(obj_ref)
```

Creating an iterator performs no query work. Its first `next()` retains producer
topology, candidate knowledge, required metadata and authority proofs using the
normal terminal retry/refresh rules. Later Store/cache mutations are observed by
a new iterator, not spliced into an already-started one. This remains one cut per
contributing Store, not a global federated transaction. Iterator matching and
relationship traversal advance with consumer demand.

Candidate inventories and metadata can still require complete capture before
the first value. Verification admits the complete detached candidate domain, and
metadata predicates validate their eligible domain before output, just as with
scalar terminals. Algebra, closure and owner/target projections may need fixed
intermediate memberships. Consequently, this API avoids a final result collection
but does not promise bounded-memory or one-record-at-a-time Store IO. It preserves
existing scan/index controls rather than introducing broader indexed coverage.

No Store fence or SQLite reader spans a yield, so a paused consumer does not hold
back Store writers. Exhaustion, error or explicit `.close()` releases the
iterator's retained execution. A `break` alone does not close a retained generator;
use `contextlib.closing` when consumption may stop early:

```python
from contextlib import closing
from itertools import islice

with closing(repo.query().object_refs().iter()) as refs:
    for obj_ref in islice(refs, 10):
        print(obj_ref)
```

Advance a started iterator only in the process and thread that first advanced it;
other consumers receive `QueryError` before further evaluation. Independent
iterators may run concurrently. Pre-output source recapture uses normal retries;
a recapture after a value was delivered raises `QueryError` instead of restarting
and replaying a prefix. Other late failures and interruptions propagate and end
that iterator; a delivered prefix is not a successful complete-query result.

`OccurrenceQuery.iter()` yields typed occurrences with raw occurrence caps intact.
It walks roots in identity order and paths in deterministic depth-first
relationship order. This avoids globally collecting/sorting every matching path;
its traversal order need not match `OccurrenceSet`'s global ordering (for example
in owner-digest collision buckets). Necessary root and target-metadata validation
still precedes output. A zero raw-occurrence cap performs no source work.

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
Fixed occurrence refinements retain their captured paths while reading explicit
metadata scopes afresh at each terminal; those scopes remain subject to refresh,
scan, and coverage controls without becoming discovery producers.

## Refresh Lifecycle And Scan Controls

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
propagate; absent optional backends create no maintenance target and trigger
neither maintenance scan admission nor a warning. A direct exact positive can
therefore run without an inventory scan under `"forbid"` or `"warn"` even with
`True` or `"always"`. When the target
becomes available, its explicit cycle still requires scan admission before work.
Memory and disabled Store policies have no persistent refresh obligation.

```python
universe = repo.query().refresh(True)
universe.cdefs().count()          # fulfills this request's current targets
universe.state_refs().collect()  # fresh evaluation, no repeat explicit cycle
universe.refresh(True).count()   # fresh request over the effective expression
```

Algebra preserves each operand's producer/scope bindings: combining independently
completed A and B universes does not refresh A for B's request or B for A's request.
This also holds for independent requests derived from one reusable base query:
sharing the base's source does not attach either request to its sibling's scopes.
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
Owner exceptions propagate unchanged; failed/interrupted-owner waiters receive
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
Independently maintaining multiple destinations for the same invalidation domain
across clients is unsupported: shared consumable dirty tokens cannot acknowledge
each destination independently. Configure one destination consistently across clients.
A Store attached after topology retention joins the next terminal. Inherited Repo
topology after fork raises `RuntimeError` before the parent-process configuration
lock; open a fresh Repo in the child. Query does not reset Repo-owned locks.
Request synchronization resets by PID before acquisition; inherited derivatives
share pending child request state. Direct Store execution remains subject to existing
Store, import, and SQLite lifecycle rules and uses fresh views. Requests are ephemeral
and process-local; there is no new serialization, transport, or deepcopy contract.

Refresh and scan/index policies are terminal-owned controls: they apply to
producers **and** external stored, alias, and metadata scopes, algebra operands,
closure, occurrence targets, and owner/target projections. A fixed producer is
source-free only when its restrictions are source-free too. Projections inherit
controls without inheriting raw occurrence caps. An explicit control on the outer query governs its nested
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
