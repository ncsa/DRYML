# Repos and Stores

`Repo` coordinates live Object realizations, saved Node persistence, and one
or more connected Stores. A Repo is local process authority, not a transport or
security boundary. Store data, definitions, symbols, and serialized payloads are
trusted DRYML inputs; supported shared use is same-host use on a local filesystem
with the documented locking, atomic-replace, and SQLite semantics. Distributed
filesystems and cross-host coordination are unsupported.

## Current Format-4 Store Primitives And Consumer Boundary

New built-in Stores use format 4 with a persistent Store UUID and authenticated
identity-3 Node metadata. Existing format-3 Stores are rejected intact; there is
no automatic conversion. The Store-owned APIs currently available are:

- `install_creation(root)` / `install_creations(roots)`: install normalized
  construction metadata, returning creation digests. A batch shares one Node
  table; a header is neither an independently published recipe nor available state.
- `install_state(root, staging, codec=..., codec_version=..., interpretation=...)`:
  authenticate an owned `create_local_state_staging()` allocation's private `data/`
  and install/reuse its full StateKey variant. The receipt contains detached
  `state`, `installed`, and `durability`. Same-CDef/hash captures reuse one directory;
  equal codec content in different CDef buckets is independent. On failure,
  preserve authoritative metadata and inspectable partial work; an unconsumed
  staging allocation can be explicitly discarded by its allocating Store. A complete
  Store-prepared stage left by intent failure can be retried with the same exact
  creation/codec input without capturing its payload again.
- `read_creation(cdef_digest)` / `inspect_states(cdef_digest, available_only=False)`:
  read detached creation/history metadata without Objects or payload IO.
  Inspection interprets one bucket once within a coherent batch. It is a complete
   single-bucket inventory; use `query().local_states(origin=...)` for indexed paging.
- `read_state(key, destination=None, codec=None, codec_version=None)`: context-managed fresh validated
  data source. When supplied together, destination/codec/positive codec version are admitted before
  payload IO. Only `source.data_directory` is lent to a codec; framework metadata
  siblings are outside it. Context exit or Store closure expires the source.
   Detached metadata remains inspectable. A native shared full-key lease spans
   fresh validation and borrowing, preventing cooperating cleanup until context
   exit. A read creates no durable retained root.
- `publish_node(root, kind="recipe"|"saved")` / `read_node(digest, kind=...)`:
  explicit independent metadata publication/read. Saved roots require complete
  facets. Required ObjectId anchors compare canonical projections, so state-pin
  changes can contribute multiple saved roots to one declaration. Configuration
  conflicts, malformed evidence and missing established anchors fail explicitly.
  Publishing saved metadata does not automatically publish a recipe root.

Supported mutations are journaled under one Store authority cut and synchronously
maintain enabled SQLite schema-11/coverage-3 projections before complete success.
DirStore also supplies physical grants and active-use leases as described below.
Ordinary Repo saves now use the format-4 Node/state-bucket pipeline described
below. Node build/load/apply/restore use full-key state readers. Managed lifecycle
remains a separate consumer boundary. Repo/Store queries use the current Node
discovery consumer; legacy snapshot/declaration writers and standalone Query V3
read views are explicitly refused on Store 4. `none`/`memory` have no persistent-maintenance obligation; `auto` uses
usable SQLite and declares nonpersistent fallback only for optional-backend
unavailability. Explicit `sqlite` never falls back. One persistent destination
and compatible settings are bound per Store authority domain before mutation;
conflicting enabled handles are refused. `store.mutation_index_disposition`
reports policy, not readiness. The remaining
historical consumer appendix records retired interfaces, not another supported
runtime or a conversion path. Snapshot-directory workflows are not supported
against built-in format-4 Stores.
New consumer preflight must declare `record_family="dryml-node-graph/v1"`; the
operation's human-readable diagnostic name does not select a persistent format.
Legacy publication entry points remain explicit `StoreCapabilityError` refusals:
definition/declaration/claim records, old aliases/main references, V3 snapshots,
and old reference annotation/lineage writers install nothing. They do not provide
a Store migration path. This does not reject ordinary legacy reference values at
their separately supported explicit transport/value-codec boundaries, or change
current Node queries and buffered archive metadata publication.

DirStore installation primitives return after native filesystem content and
directory persistence barriers. ZipStore's corresponding receipt is explicitly
buffered; only successful archive `commit()` establishes durable completion.
Failed commits propagate and retain dirty evidence; closing discards the buffered
view. File-like ZipStores permit v4 reads but refuse writable publication.

After interrupted publication, `recover_authority()` records the exact visible cut
and retries unresolved native persistence barriers before acknowledgement or
matching-state reuse. A visible variant alone is not a durable installation.
Repair touches affected metadata/parent entries without capture, restore or body
execution and never executes missing transitions. Repeated barrier failure raises
`StoreMutationError` with its mutation/cut receipt and preserves pending evidence;
SQLite maintenance or automatic query refresh cannot bypass this gate. Unsupported
standalone directory barriers fail explicitly rather than infer durability from
visibility. See [Formats](formats.md) for mutation-intent v2 and backend limits.

### Ordinary Save, Explicit Reuse And Routing

`Repo.save_object(obj, states=None, ...)`, `Repo.save(obj, states=None, ...)`,
`obj.save(...)` and the global `save_object(...)` return a **SavedNode**: the same
immutable ConcreteNode carrier with complete required identities and local-state
facets. `saved.at(path).state` is a detached `LocalState` carrying its full
`StateKey`. `saved.state_spec()` is a rooted, alias-preserving reusable selection
map. There is no separately authoritative StateRef state map in this save path.

```python
first = repo.save(obj)
later = repo.save(obj)                      # fresh capture, even if unchanged
reused = repo.save(obj, states=first.state_spec())  # no selected-node capture
description = repo.describe(obj)            # retained construction/identity only
```

`states` also accepts a partial mapping from typed GraphPaths to LocalStates.
Unspecified distinct live stateful nodes capture **once** per logical save;
shared aliases do not capture twice and independent equal nodes do not merge.
Selected states are checked for creation/configuration/codec compatibility and
current source availability before any capture hook. Missing selections fail
explicitly. `source_store` selects a connected source; `source_stores` accepts
full-StateKey-to-connected-Store overrides. Same-bucket source batches interpret
their header once at a validated cut. A previous receipt is historical evidence,
never automatic permission to reuse mutable live state. `deep_capture` remains an
accepted spelling; ordinary saves already capture every unselected node.

Before hooks, save also freshly validates any destination capture association
named by the live root's previous saved receipt. Malformed installed metadata
raises `RepoSaveError` with its authority-error cause, preserving the bytes and
running neither capture nor environment observation. A missing destination
association permits a new save. This check does not certify current private
state or turn a previous receipt into permission to reuse it.

Capture receives only an empty codec-private staging `data/`. Content identity
includes codec version, interpretation, inventory and bytes. An unchanged capture
can discover the same full key and reuse its installed directory without copying
it; explicit same-Store reuse needs no staging copy. Equal content in different
creation buckets remains separate. New saved identity/state facets do not alter
retained construction pins, and automatically allocated IDs do not become
retroactive construction constraints.

Routing retains one configuration/source view for the entire save. `store=`
selects one whole-graph closure; `SaveRouting` preserves first/all matching and
per-object/closure placement. Per-object placement publishes compact child saved
roots at their destinations, without copying child bytes into parent snapshots.
Resolved dependencies publish first where possible. An acyclic live Object graph
may induce a cycle between whole-Store batches, including A → B → A. Prepared
local batches can publish in that case: all target keys are leased together and
each batch protects its own physical resources before local membership. This does
not yet certify cross-Store recovery. A target blocked by failed providers, or
transitively dependent on such a target, remains **unattempted**; independent
targets continue. Recovery sets and live receipts qualify only after all required
full keys have confirmed available, physically protected providers.

One shared Node table and compact saved/recipe/creation/projection views are
prepared for the save. Each destination batches installed variants, physical
grants, declaration anchors, memberships and names through its Store journal.
`alias=` names the saved graph and `main=True` names the independently published
recipe root in that same batch. Ready enabled indexes perform one coalesced SQL
projection commit before complete return. An identical saved graph with the same
existing user root/names is an authoritative no-op: another plain capture still
runs, but no authority epoch or SQL write is added. Disabled/memory indexing
creates no persistent sidecar. Initial/stale-index maintenance is separate setup
work, admitted before capture.

Ordinary saves currently require the durable DirStore retained-batch seam.
ZipStore still supports buffered low-level primitives, but its lack of durable
owner grants makes ordinary retained save explicitly unsupported **before
capture**. Ordinary saves support requested annotations and passive creation,
environment, requirement and first-capture evidence through closed Node metadata
records in the same journal/index batch. Managed operation lifecycle policy remains
separate. These boundaries do not add Store-v3 migration.

### Build, Load, Partial Apply And Full Restore

`repo.build(concrete_node)` requires prior concretization. The convenience
`recipe.build(repo=repo, identities=..., states=...)` composes constraints before
the same concretization boundary. Build allocates missing ObjectIds only and
initializes unspecified state. Selected LocalStates are restored once after all
associations and current source payloads preflight.

To establish stable identities before dispatching a recipe to workers, call
`concrete_node.allocate_identities()` once in the coordinator and reuse the returned
graph. This [metadata-only allocation](objects_and_defs.md#allocating-identities-before-dispatch)
fills missing owned stateful IDs without a Repo, Store declaration, construction,
or state capture. Optional `check_unique=True, repo=repo` checks new IDs against
connected declaration anchors and known live/cached metadata, retrying collisions
without publishing reservations. The default `check_unique=False` performs no
Repo/Store reads. Both modes preserve selected states and do not reserve exclusive
execution; managed operation ownership remains a separate boundary.

`repo.load(saved_node)` / `repo.load_state_ref(saved_node, reuse_live=...)` require
saved completeness, then freshly validate full-key availability. They do not
initialize omitted state or substitute another checkpoint. Stateless saved graphs
are valid without captures. Reused eligible live nodes run their selected hooks;
historical receipts never establish current private state.
Matching can select a unique whole cached topology even when its stateless Method
dependencies have no independent cache entries. It still restores selected state;
ambiguity constructs fresh. Fresh loads attach only the requested root receipt,
while reused children preserve any independently established historical receipt.
Every missing full key is reported before construction or live-node reservation;
malformed selected capture/declaration metadata remains an explicit authority error.

`repo.apply_state(obj, states)` applies only the supplied StateSpec/typed path map,
preserving unspecified state, identities and sharing. It captures/publishes nothing.
`repo.restore(obj, saved_node)` checks the **original supplied** canonical object
projection, restores dependency-first, then adopts supplied construction pins with
live ObjectIds unchanged. Subsequent captures use those adopted creation views.
State selections and lineage destinations are matched to the receiver's exact
occurrences and ObjectIds, including independently decoded recipe-equal set
members. A `SetMember` ordinal addresses its own graph view; it is not portable
identity evidence between saved and live graphs.
Foreign compatible state retains the donor full StateKey rather than being rehomed.

Both operations accept `source_store` and `source_stores` full-StateKey overrides
within the connected Store scope. Distinct selected keys are read in grouped Store
batches, without scanning sibling payloads. Each destination independently checks
creation-CDef and codec compatibility. Native shared full-key leases protect all
borrowed data through hooks; no Store writer fence is held during user code.

`repo.prepare_restore(obj, saved_node, reservation=...)` exposes a one-shot
owner-thread handoff for another Core caller. Its context retains source leases
and the supplied/new live reservation. `apply()` consumes it once and returns the
supplied saved graph; `close()` releases unused preparation without hooks.

Pre-hook validation failures leave usable live graphs usable. After any hook begins,
a failure invalidates the affected graph for framework state IO and evicts it from
caches. Recovery requires fresh materialization from a valid available SavedNode,
for example `repo.load_state_ref(saved_node, reuse_live="never")`. No rollback is
claimed. See [Objects and Definitions](objects_and_defs.md#restore-and-references).

### Retained User Roots And Partial Save Reports

A normal save creates a user-owned retained graph root. Each target's exact
owner intent names its full keys and staging resources; physical provisional
grants precede local graph membership and lease release. A complete target-local
user association adopts the intent; it does not assert external payload
availability. Repeated identical publication reuses the adopted user-root
token; a new attempt never resurrects an abandoned/retired token. The Core batch
seam also admits provisional publication for a later lifecycle owner, without
creating an adopted permanent user-save reason. Managed association policy is
not implemented by this seam.

`report_stores=True` returns `(saved, StoreReport)`. `RepoSaveError.report`
preserves partial cross-Store outcomes. Each combined batch publication reports
Store UUID, authority cut, mutation UUID, index disposition, retention token,
confirmed available `state_keys`, their physically protected subset
`protected_state_keys`, and `planned_state_keys`. An available key without the
matching owner intent and physical grant does not qualify recovery. `state_ref` is the
historical field name for its planned SavedNode; `snapshots` lists independently
confirmed saved graph metadata, which can outlive payload availability. Each
snapshot's `required_stores` is a sufficient protected recovery set at
reconciliation, or empty for an incomplete closure. The report's top-level
`required_stores` applies to the root graph. Available but unprotected bytes remain
visible in `state_keys`/`state_stores` without becoming a complete recovery claim.
A confirmed graph can survive index/replica failure and become `obj.last_state_ref`
only when its entire closure has qualified protected providers. An incomplete root
preserves the previous receipt. Reconciliation and receipt updates run before the
operation releases its complete target-key lease set; uncertain grants are not
retired on error. None of these facts turns partial or uncertain work into
successful completion. Cancellation retains its original exception identity and
partial report, including when an earlier independent target already failed;
remaining targets are unattempted.

An unrelated cooperating writer can invalidate a detached read-back cut after
the save has published. Save confirmation retries that metadata observation at
most eight times, without repeating capture, state hooks, or publication. Every
attempt validates a fresh cut; invalid authority is not retried as contention.
Exhausted read-back remains an unconfirmed save failure with its partial report,
and does not advance the live root's receipt.

Repair a failed index with `store.open_query_index().maintain_store()`. For an
authority-partial save, reconnect the exact Store and call
`store.repair_save_batch(publication.mutation_id, publication.retention_token)`.
Repair uses the persisted journal and owned staging, never another capture.
Index maintenance (including automatic query refresh) may acknowledge a partial
projection first; that does not consume the original mutation UUID's repair
eligibility. Missing transitions are a new journaled publication, while the
original receipt retains its actual classified before/after/projection evidence.
An exact completed repeat makes no authority changes.
It serializes on the owner token, rejects abandoned/retired intents, acquires
independent nonblocking exclusive key leases and refuses missing staging or a
conflicting later cut. An uncertain grant remains protective until owner-fenced
resolution. To retire a completed ordinary user root, release each target-local
token from its report with `repo.release(token, owner_store=publication.store)`.
The Repo releases protection then attempts affected-key reclamation. The lower-level
`Store.retention.release` releases grants only. Provisional roots require conclusive abandonment
before release. Withdrawing/reinstalling an identical key uses a separate
exclusive phase, never an upgrade of held shared leases.

### Inventory, Explicit Retention And Federated Cleanup

`repo.inspect_states(cdef_digest, available_only=False)` returns detached
`StateInventoryEntry(domain, descriptor)` rows for one creation bucket across
connected Stores. It reads metadata/presence only, constructs no Objects and reads
no payload bytes. `descriptor.state.key` is the compound StateKey;
`descriptor.available` distinguishes historical knowledge from published payloads.
For indexed, bounded discovery use `repo.query().local_states(origin=...)`.

`repo.retain(saved, owner_store=None)` validates an existing complete SavedNode's
selected provider closure and returns an independent `RetentionIntentRecord` root
handle. It captures/restores/serializes no private state. The durable owner defaults
to `repo.default_store`. Keep that handle (owner UUID, token and affected resources)
for explicit release; references alone do not grant retention. Ordinary saves already
create their own independent user roots, so retaining again adds another reason.

```python
saved, publications = repo.save_object(obj, report_stores=True)
extra_root = repo.retain(saved)
for publication in publications.publications:
    outcomes = repo.release(publication.retention_token, owner_store=publication.store)
    # Still retained by extra_root (and any other independent roots).
outcomes = repo.release(extra_root)  # only the affected Store/full-key closure

inventory = repo.inspect_states(saved.state.key.cdef_digest)
outcomes = repo.delete_states([entry.descriptor.state.key for entry in inventory])
```

`repo.delete_states(keys, store=None)` attempts every requested full key in each
connected Store, or in the selected connected Store. `release(root, owner_store=None)`
first retires the named owner's exact grants, then attempts only acknowledged
resources. Both return tuples of `StateCleanupOutcome`: `domain`, `key`, `phase`,
`status`, optional Core `outcome`, and optional `error`. Statuses include `removed`,
`retained`, `in-use`, `unavailable`, `obsolete`, `cleanup-pending`, `failed`, and
`unattempted`. Reports describe individual targets, never a distributed rollback or
one global success. A removes/B fails/C continues when C is independently safe;
disconnected targets and targets lacking prerequisite acknowledgements are unattempted.
Root-retirement grant/owner/index failures raise with `.cleanup_outcomes` attached,
including completed independent work. Deletion-phase errors are failed outcomes
in both APIs. Direct deletion reports ordinary per-target failures;
interruption raises with completed/failed/unattempted evidence.

Deletion is nonblocking with respect to active use: it neither waits nor queues a
deferred request. Release from within the same caller's active key scope returns
`in-use` **without retiring the root**; explicitly retry release after that scope
ends. Other readers keep protecting data after a root is retired. Shared publication/
reader leases end before the separate exclusive deletion phase; no lock upgrade occurs.
Physical garbage failure alone returns `cleanup-pending` after valid withdrawal/index
updates. Use the matching Store's `retry_state_removal(key, outcome.incarnation)` for
exact retry; a newer incarnation makes the old public-data deletion obsolete.

Cleanup preserves saved graph/definition/manifest/name history and ordinary managed
results. A later `repo.load(saved)` fails if its payload is unavailable; it does not
invent state or reexecute completed work. Enabled index failures remain failures,
and rebuilding an index never deletes historical authority to make it fit.

### Physical Retention And Active Use

Protection is scoped to the Store authority domain **and full StateKey**, never
just the codec hash or the creation bucket. Historical graph/manifest references
alone retain no private bytes. `install_state()` is a low-level unretained
installation: its shared publication lease ends on return. A caller must confirm
protection before exposing a retained association; if cleanup wins before grant
acquisition, protection fails as unavailable rather than claiming a usable root.

`DirStore.retention` exposes Core handshake primitives. The owner Store can be
different from the Stores holding the bytes:

```python
from uuid import uuid4

token = str(uuid4())  # persist this stable root/intent token in caller authority
owner.retention.prepare(token, saved_graph_digest, ((payload_store, (key,)),))
owner.retention.protect(token, (payload_store,))
owner.retention.adopt(token, (payload_store,))
# The caller can now confirm its association, including token, in its own authority.
# Only after the new association is confirmed may it retire an old root:
owner.retention.release(old_token, old_payload_stores)
```

These methods implement evidence primitives, not managed role/association policy.
Each independent user/durable/recovery reason uses its own token. The authoritative
owner ledger records owner UUID, stable token, graph digest, exact target Store
UUIDs/full keys, named staging resources, and disposition. Per-root grants live at
`states/<shard>/<cdef>/<codec-hash>/grants/<owner-root-digest>.record`, beside the
protected bytes. Deletion inspects these actual entries, never reverse counts or
an index. Provisional and adopted grants protect equally.

- `prepare(token, graph_digest, targets, staging=())` durably names affected work
  before any grant. Targets are `(Store, iterable[StateKey])` pairs. Optional
  staging entries are `(allocating_store, staging_handle)` pairs, recording exact
  allocation IDs and owner markers. Same-token retries require identical facts;
  a new attempt requires a new token.
- `protect(token, connected_stores)` may incrementally protect prepared targets.
  `adopt(token, connected_stores)` requires every named target and grant. Both
  serialize with owner dispositions; neither establishes a managed association.
- `abandon(token)` is an explicit owner-fenced tombstone for prepared work. It
  permanently rejects later protection/adoption. Death, elapsed time, missing
  index entries and disconnected/unreadable owners never call it automatically.
- `release(token, connected_stores)` requires adoption or prior abandonment,
  persists `release-pending` **before** removing owned grants, and acknowledges
  each target independently. A failed target does not prevent independent targets
  from completing; the first raised error carries `retention_failures` and
  `retention_intent`. Owner/control errors or interruption stop dependent work.
  Disconnected targets or live staging return an explicitly `release-pending`
  record. Only confirmation of all targets produces `released`.
- `read(token)` reads one exact ledger record. `pending(after=None, limit=100)`
  pages the affected pending ledger, including interrupted journal work, without
  scanning operation history. A zero limit performs no source work. Reconnect
  the exact targets and retry the named token; do not infer release from absence.

The pending-resource head is persisted before a new disposition. Settled tokens
leave that head only after disposition publication and enabled-index
acknowledgement, in a separate mutation. An interruption can leave extra named
work, never an unnamed authorized release. Exact `replay_mutation()` repairs an
interrupted owner-ledger write. Grant publication/removal instead requires the
freshly connected owner's `protect()`/`release()` handshake; generic replay cannot
resurrect an abandoned grant.

Staging allocations have exact owner markers and native lifetime leases, bound to
the allocating handle/thread. Foreign live allocations cannot be published,
discarded or consumed by replay. `release_local_state_staging(handle)` explicitly
releases the allocation lease while preserving recovery work; it is not
abandonment evidence. Closing its owner also releases staging leases but preserves
the allocations. Owner-fenced cleanup checks the exact marker and nonblocking
live lease before removing only ledger-named work. It never inventories a shared
staging directory to guess what is abandoned.

`delete_state(key)` tries an exclusive native lease **nonblockingly**, before
Store writer fences. It returns `StateRemovalOutcome`:

| Status | Meaning |
|---|---|
| `retained` | Authoritative grants prevent reclamation. |
| `in-use` | An active reader/publication lease or unresolved publication prevents reclamation. |
| `unavailable` | No state variant exists. |
| `removed` | Availability was withdrawn and private data removed; metadata history remains. |
| `cleanup-pending` | Availability withdrawal completed, but physical garbage removal raised `cleanup_error`. |
| `obsolete` | A newer incarnation superseded the requested old removal; current data was untouched. |

Retained/in-use results neither wait for protection to end nor queue a deletion
request. Retry explicitly after releasing owned roots/leases. Index, grant,
control and availability-journal failures propagate as failure/ambiguity; only
physical garbage deletion has the separate cleanup-report outcome.

Withdrawal journals the exact availability incarnation and a removal record at
`.../removals/<incarnation>.record`. The exclusive key lease lasts through physical
removal. `retry_state_removal(key, incarnation)` reacquires it and rechecks grants,
pending publication and the current incarnation. New same-key publication makes
old public-path removal obsolete. Republishing withdrawn/partially removed data
is a separate nonblocking exclusive recovery phase, with a complete validated
owned stage. Old private leftovers are detached to incarnation-named remnants;
the canonical bucket header, manifest and graph history remain. Old retries may
remove only those retired remnants, never the new public `data/` directory.

All grant, owner-ledger and availability changes enter the Store journal and
enabled schema-11/coverage-3 index. The `retention` relation exposes per-key grant
postings. Requested metadata has its own supported closed relation; managed
lifecycle associations remain explicitly unsupported. Missing reverse indexes may be rebuilt from authoritative records;
they never authorize removal. Low-level `apply_prepared_mutation()` acquires key
leases for state installations and refuses manual retention/availability
transitions in favor of these owning primitives.

ZipStore supports **transaction-local** active leases and buffered cleanup only.
Different extraction handles are independent views; their leases do not protect
one another's buffered bytes or establish cross-archive grant authority. Durable
`retention` ownership/grants are explicitly refused, and read-only file-like
archives refuse cleanup. Closing a handle expires borrowed sources immediately
and defers extraction removal until its last active key lease/writer exits.

For several keys, prepare the complete set and use `store.state_leases(keys)` or
the plan adapter `repo_plan.state_use_leases((store, key) pairs)`, then freshly
validate sources while protected. Growing a held set is refused: release,
reacquire in global order and revalidate. Do not upgrade an owned shared lease to
exclusive. See [Advisory Locking](locking.md) for native support and lock order.

### Publication, Names And Interrupted Work

- `authority_cut()` returns the persistent Store UUID, committed epoch and exact
  pending mutation identities. It does not certify query-index coverage.
- `compare_and_set_node_name(NodeNameRecord(name, kind, digest), expected=...)`
  assigns a format-4 name to an independently published recipe/saved root. Use
  `expected=None` for absence or the previously read exact record for replacement.
  `read_node_name(name)` validates its target membership. These names are separate
  from the unmigrated ObjectRef/StateRef alias and main-reference APIs.
- `delete_node_name(name, expected=...)` and
  `rename_node_name(name, new_name, expected=..., expected_target=None)` use exact
  source/target CAS. Mismatches raise `StoreAliasConflictError` before effects.
  Interrupted rename may retain both complete mappings; the journal records that
  actual result instead of claiming a rollback.
- `withdraw_node(digest, kind="recipe"|"saved")` removes only that root's explicit
  membership and declaration-source contributions. It returns False for an absent
  or still-named root. It preserves immutable graph metadata, canonical declaration
  history, manifests and all payloads; this is not reclamation or retention.

Graph/codec validation happens outside writer-blocking fences. Short metadata scopes
capture/revalidate the cut and exact byte/directory witnesses. An unstable read
raises `StoreCutChanged`; publication retries stale preparation within a bounded
budget. Do not surround these high-level APIs with an externally retained
`writer_lock()` or `authority_read_fence()`. Prepared byte/association checks,
installs, cuts, the short prepared SQL transaction and exact acknowledgement
belong inside those scopes. Shared use requires same-host native
advisory locks, atomic rename and the filesystem persistence guarantees documented
in [Filesystem](filesystem.md); unsupported network/distributed filesystems are
outside this contract.

Intent and invalidation persist before query-visible effects. After a visible
failure, `StoreMutationError` exposes `mutation_id`, `cut`, `confirmed_refs`, and
`index_disposition="repair_pending"`, and `recovery_attempt`, with the original cause. The epoch describes
the honest classified result, not all-or-nothing completion. Cancellation retains
the same evidence attributes. Preserve partial staging when recovery is needed.

Open a compatible Store and call `recover_authority()` to classify interrupted
work. This reads affected metadata only; it neither runs capture/restore/hooks nor
deletes authority. `pending_mutations()` supplies exact expected transitions and
observed projection bytes for index consumers. `replay_mutation(uuid)` resolves
the exact persisted classified intent independently of that pending list: index
maintenance or automatic query refresh may already have acknowledged its partial
projection without completing its intended authority transitions. Repair uses the
original before/after transitions, classified byte witnesses and surviving owned
staging without graph preparation. Missing transitions publish in a separately
journaled mutation, with enabled index completion for that new cut; an already
acknowledged original receipt is neither rewritten nor acknowledged again using
an obsolete SQLite receipt. Pending original work is acknowledged only at its
captured cut. A later Store epoch, changed byte witnesses or missing
staging fails explicitly when transitions remain incomplete; replay does not
guess through ABA changes. A repeated repair of exact final records (including
availability incarnation) is a no-op. Already visible
records still cross native persistence barriers before acknowledgement.
An earlier repair never acknowledges a later writer. Missing/corrupt/future
authority or intent versions are errors, not permission to reset the Store.

The Store-owned `PreparedMutation`/`apply_prepared_mutation` seam accepts detached
immutable byte facts from the same handle, with cut/CAS witnesses and affected
scopes. It returns `(mutation_uuid, cut)` after enabled persistent projection and
exact acknowledgement complete. In nonpersistent mode this explicit low-level
seam leaves work pending for its owner; ordinary public mutations acknowledge
their nonpersistent disposition synchronously. `acknowledge_mutation(uuid, epoch)`
is only valid after that exact work and its coverage are incorporated, and requires
an exact SQLite receipt when enabled. Authoritative no-ops with a current index
churn neither epochs/intents nor SQL rows. Missing/stale enabled sidecars still
require visible metadata-only replay/rebuild or refusal. Repair changes no Store
format or payload bytes and never repeats application capture. See
[Query Index Backend Contracts](query_index_backend_contracts.md) for relation
coverage, failure boundaries and the maintenance-free index-level read seam.

ZipStore buffers these same cuts and intents in its extraction.
`committed_archive_cut` records the last confirmed durable archive cut; it remains
unchanged after a failed commit. `ZipStoreCommitError` includes the buffered cut,
confirmed references and previous durable cut with `durability="buffered"`.
Failure before replacement leaves a reopened archive on its previous contents;
failure after replacement visibility is ambiguous, not rolled back. The live dirty
extraction and stable attempted-archive evidence remain retryable. Only a successful
native archive persistence boundary establishes a new confirmed durable cut.

See [Formats](formats.md#store-format-4-cdef-owned-state-and-metadata-readers) for
record families, physical layout and failure rules. Repo and managed integration
must consume these Store APIs rather than form or mutate bucket paths.

## Historical Store-3 Consumer Appendix — Unsupported

The discussion below preserves superseded implementation evidence. Its StateRef
snapshot, claim/declaration, deferred-validation-cache, alias and fork workflows
are **not current format-4 instructions** and do not authorize conversion or a
parallel runtime. Use the current Node sections above, [Managed Operations](managed_operations.md),
and [integrated verification](object_state_lifecycle_verification.md). Current
StateVariantManifest v1 authenticates all inventoried bytes at use; a codec's
logical deferred-file annotation does not enable the old deferred-hash proof.

### Historical Saving With Routing

`Repo.save_object()`, `Repo.save()`, `Object.save()`, and module-level
`save_object()` publish one immutable root `StateRef`. Their shared keywords are
`main`, `store`, `alias`, `deep_capture`, `match_mode`, `graph_mode`, and
`report_stores`, plus metadata `annotations`, `source_store`, and `source_stores`.
The removed ordinary-save `federated` keyword is not accepted.
Fork and query federation are separate APIs and retain their own behavior.

`SaveRouting` is an immutable ordered sequence of `(Selector, Store)` bindings.
Rules only refer to already connected Store handles; configuration neither opens
nor creates a Store. `match_mode="first"` selects the first matching rule, while
`match_mode="all"` selects every distinct matching Store as required replicas.
An unmatched Object uses `Repo.default_store`; a selected unavailable or unusable
Store fails the save rather than falling through or being created implicitly.

```python
from dryml.core import Repo, SaveRouting, Selector
from dryml.core.store.dir import DirStore

experiments = DirStore("./experiments")
models = DirStore("./models")
repo = Repo(
    [experiments, models],
    save_routing=SaveRouting(
        ((Selector(Experiment), experiments), (Selector(Model), models)),
        match_mode="first",
        graph_mode="per-object",
    ),
)

state_ref, report = repo.save_object(experiment, report_stores=True)
```

With configured `graph_mode="per-object"`, each Object's local state is saved to
its own selected destination set. The requested root and routed children have
independent discoverable StateRefs. A child snapshot can require other connected
Stores for exact recovery, so its Store alone is not necessarily self-contained.
Each Object is captured once per save; replication publishes the same captured
state and exact identity rather than recapturing a replica.

`source_store` selects one complete existing root snapshot when replicas are
ambiguous. `source_stores` maps exact root or routed child StateRefs to their
selected connected Stores. These options select whole captured records and
payload authority; DRYML never merges evidence fields from different replicas.

With `graph_mode="closure"`, the root's selected destination set receives a
complete, independently recoverable closure. Routing of descendants does not add
external dependencies to those root replicas. An explicit `store=` always wins:
it selects one whole-graph closure destination and bypasses rule selection,
placement, and replication. Supplied mode values are still validated.

`None` mode overrides inherit the retained Repo policy. A Repo without a policy
uses first-match/default-Store closure behavior. `SaveRouting()` is instead an
enabled empty policy: unmatched Objects use the default Store with per-object
placement. `Repo(..., save_routing="per-object")` and `"closure"` create an empty
first-match policy. On an existing Repo, `set_save_routing("per-object")` or
`set_save_routing("closure")` changes only placement and preserves ordered rules
and `"all"` replication; use a complete `SaveRouting` value to change matching.
Changing a policy never moves, deletes, or synchronizes stored state.

A save retains one coherent snapshot of routing, connected Store order, and the
default Store. Later routing or Store configuration changes affect later saves,
not the active graph. Direct concurrent mutation of `repo.stores` is unsupported;
`Repo.close()` rejects while a save or managed topology lease is active.

## Publication Evidence

`report_stores=True` returns `(StateRef, StoreReport)`. `StoreReport.target_stores`
lists ordered root destinations, `state_stores` records confirmed local-state
destinations by graph path, and `required_stores` is one ordered sufficient set
of connected Stores for exact recovery. `snapshots` lists independently confirmed
root or child StateRefs with their selected snapshot and recovery Stores.
`publications` is the per-boundary ledger: definition, lineage, state, snapshot,
object/state metadata, membership, claim, alias, main, index, and commit work is marked `completed`,
`failed`, `unattempted`, or `uncertain`.

There is no cross-Store transaction. A partial publication raises `RepoSaveError`
with its immutable partial report after a plan exists. Completed immutable records
remain available for inspection and recovery; they are not deleted to simulate a
rollback. Initial declaration claims remain fenced and are reported separately
from StateRef and membership authority. Closure saves also complete declared
descendant claims after enclosing snapshot membership succeeds, in dependency
order. Their claim entries name the exact descendant StateRef projection; they
do not imply an independently published child directory. Root aliases and main references are
written only after every selected root snapshot is complete, in every selected
root Store; independently published children never inherit root names. Derived
query-index failure remains visible and does not erase authoritative records.

`Repo.save_object()` may leave a path-backed ZipStore buffered. `Repo.save()` and
temporary convenience Repos commit their configured Stores before returning and
record those commit boundaries. A successful buffered save is not an archive
commit. A completed root `last_state_ref` receipt can remain inspectable when a
later index, name, or commit boundary fails.

## Loading And References

`Repo.load(cdef)` and `load_object(cdef)` construct structural Objects and do not
infer state. `Repo.load_or_build()` may create missing structure.

`Repo.load_state_ref()` accepts either a directly published complete snapshot or
an exact descendant projection retained by a complete enclosing closure snapshot.
For an enclosing projection, exact-load preflight validates the projected
definition, ObjectIds, state hashes, lineage, and enclosing snapshot-local payload
paths before construction. Loading a projection does not synthesize or require an
independent child snapshot directory; per-object routed descendants still use
their independently published child authority.

Exact preflight uses a Repo-owned, explicit read cut: it atomically retains the
connected handle order and acquires ordered cooperating-writer authority fences
before reading. Supported default/routing changes affect later calls. Read-side
physical aliases remain permitted, but deduplicating their lock domain does not
merge distinct handles' evidence; ZipStore extractions are always distinct views.

Within an eligible built-in view, preflight validates each complete snapshot
bundle once. Eager exact planning hashes each distinct physical local payload
once, checks **every** path/definition association, and validates unrequested
local siblings even when loading a projected child. A confirmed built-in delegated
placement sentinel routes to child snapshots without discarding parent bundle,
eager sibling, or staged content evidence. Generic public-reader `KeyError` keeps
its established missing-local-placement routing meaning, while conservatively
discarding partial cut evidence and preventing content admission from that cut.
Advertised built-in local corruption and genuine custom authority errors such as
`StoreAuthorityError` fail without child/replica fallback. Metadata inspection
opens no payload bytes. Restoration uses path-specific deferred validation: inventory,
types, and sizes are checked, while only eager files are hashed. Deferred evidence
cannot authorize eager copy, fork, publication, or recovery. Custom Stores and
overridden readers/validators/adapters retain their existing uncached behavior
and capability errors.
Eager custom local-state validators keep the legacy `(directory, manifest)` call
and may return `None` after successful validation, including for non-deferred
payloads during exact restoration. ZipStore wrappers likewise omit
new preparation/cut keywords when those options are absent. Archive acceleration
checks the captured DirStore implementations beneath the wrappers as well as the
wrappers themselves; replacing a base reader uses uncached reads and preserves
its custom validation errors, including after a Repo has warmed content evidence.

Cuts are process/thread-bound and expire on exit; independent nested calls begin
fresh cuts. Internal substeps may join only by passing the same cut explicitly.
No cut spans construction or restore hooks. Exceptions discard partial evidence
and release fences/topology; foreign-thread or inherited-process unwind cannot
release the owner's resources. A forked child opens a fresh Repo. `clear_cache()`
invalidates derived evidence independently of its live-tier flags, and `close()`
rejects while a cut retains topology. There is no permanent payload-validity cache.

An exact plan's `LocalStateSource` handles retain their established borrowed Store
lifetime after the cut expires. Direct directories remain Store-owned; closing a
ZipStore removes its extracted paths. Holding a source does not prove that it is
valid for another independent operation, which always validates fresh authority.

### Bounded Snapshot And Graph Content Reuse

The same Repo-owned validation facility may retain parsed primitive content for
the snapshot's `placement.json`, `metadata.json`, and `snapshot.json` siblings.
Eligible built-in reads through explicit authority cuts, including exact preflight
and managed durable validation, use this private path. Direct public Store readers
continue their fully validating uncached path. Custom Stores, subclasses and
overridden readers keep their established authoritative fallback.

A warm hit avoids the existing canonical JSON parser and its primitive projection
preparation. **Every independent cut still reads current sibling bytes and checks
filesystem shape.** Matching includes exact source bytes, component kind, parser
bounds, schema/representation and policy versions, and live parser/preparation
implementation identities. Hashes only select buckets, with at most four candidates
per bucket. Collisions, changed context or changed bytes cannot substitute another
fact. Envelope freezing, encoding/ID verification, domain decoding, sibling/target
association, aggregate closure and live-lineage checks still run. No decoded domain
objects, references, live Objects, CDefs, annotations, resource handles, extraction
paths, or payload-validity verdicts are retained.

The same facility also retains token-neutral preparation for eligible
`DefinitionRecord` and `StateRefRecord` graphs, through the existing record and
reference owners. Exact preflight and plain built-in `Repo.has_cdef_light()` reads
pass this private interface explicitly. Each read first checks current file shape,
frame, size and trailing bytes and **freshly decodes dill**. A graph hit additionally
requires lossless fresh-content matching (types, tags, labels, order, role bits,
scalar bits and array representation), exact source bytes, and matching codec,
normalizer and decoding-policy implementation/version context. Neither a digest
nor public graph/reference equality is sufficient. Malformed data cannot be
sorted, relabelled or deduplicated into a hit.

Eligible graphs have exact `ImportRef` class symbols, primitive scalar/container
and link tags, canonical primitive mapping keys, and standard non-object numeric
or bool ndarrays (including exact `FrozenNDArray`). Arrays are retained as owned
immutable bytes with dtype/shape/representation descriptors. SourceSpec classes,
quotations, embedded ObjectRef/StateRef leaves, arbitrary atoms/subclasses and
object/extension dtypes fall back to ordinary complete decoding. In an ordinary
small mean Fold exact preflight, the separately stored `MeanInitial`, `MeanUpdate`
and `MeanFinalize` records benefit; the Ref-bearing Fold root containing unsupported
source data follows the full path.

Warm eligible reads omit repeated canonical label/order proof and graph topology,
hash and role derivation inside the prepared record path. Existing codec node
construction still allocates fresh private tokens, ImportRefs, containers and
arrays, preserving sharing within one graph but never across independent decodes.
Current path/ObjectId/state/target associations and materializing class abstractness
and recorded-role admission remain mandatory. Prepared topology comparison still
derives the current target's topology; there is no universal live-definition
equality cache. Whole snapshot environment/requirements envelopes still undergo
freezing, encoding/ID verification and domain decoding. Fresh matching, hashing,
realization, association checks and payload validation remain size-dependent work.

The internal LRU admits at most **256 entries, 32 MiB of accounted owned data, and
2 MiB per entry**. Resident facts and pending reservations share those limits
across threads, independent nested cuts and internal borrowers. Charges include
keys/implementation weak identities, source bytes, prospective immutable
containers and leaves, and a conservative per-entry bookkeeping surcharge. Repeated
immutable leaves are deliberately overcounted. Budget is reserved **before** optional
retention copies; failed admission uses ordinary uncached processing and does not
reject input. Necessary decoding, detached public returns and cut-local evidence
have independent input-dependent working memory: these bounds are not a process
memory ceiling. Eviction causes recomputation, never loss of Store authority.

Facts are staged until the complete owning cut succeeds, including aggregate and
live-lineage validation. Failures/absences are never cached; a failed scope drops
pending facts and partial authority evidence. Hits rebuild detached primitive
containers and existing owners decode fresh public graphs/metadata, so mutating
nested returned lists/maps cannot affect retained content or later reads.

`clear_cache(strong=..., weak=...)` keeps the existing live-tier flag meanings and
also drops content facts and advances a shared epoch. An in-flight optional copy
keeps its budget reservation until preparation finishes, but cannot repopulate the
cleared epoch. Accepted owner close clears facts before cleanup; a failed close may
leave the Repo open with empty facts for retry. PID/closing checks precede evidence
lock acquisition, and cuts remain thread-bound. Cache locking covers bookkeeping
only; reading, parsing, preparation, matching, decoding and cleanup run outside it.

Managed durable reopen creates a temporary Repo that **explicitly borrows** the
originating Repo's facility while retaining its lifetime through a topology lease.
It owns a distinct authority cut over freshly reopened Stores; buffered validity
never authorizes the reopened view. Closing that borrower detaches without clearing
the owner's facts. Owner clear invalidates the shared epoch for every borrower,
and owner close rejects while a borrower retains its lifetime. No reopened path or
resource is stored in a content fact.

## Current Query Sources

`Repo.query()` creates one unevaluated identity universe over connected Store
authority and retained cache knowledge. `Store.query()` creates the equivalent
single-Store universe. Use `sel()` for structural or exact identity restrictions,
`cdefs()`/`object_refs()`/`saved()` for graph domains, `local_states()` for full-key
variant discovery, and `collect()` or a
scalar terminal to evaluate. Query terminals never construct Objects or open
payloads. Pass a `cdefs()` recipe result to `repo.build(concrete_recipe_node)` for
construction, allocating missing identities and initializing unspecified state.
Use `repo.load(saved_node)` or `repo.load_state_ref(saved_node)` only for complete
SavedNodes, for example from `saved()` / `state_refs()`; loading requires every
owned stateful occurrence's identity and state and never initializes omitted state.
Source graph results are Nodes; use `graph_equal()` for semantic equality
after a fresh decode. See [Graph Querying](graph_querying.md) for current domains,
bounded cuts, continuation failures, and observable repair/scan/refusal policy.
`take(0)` reads no source. Exact full-StateKey hit/miss opens no index. Ready
selective queries use bounded candidate pages, release fences before preparation
and output, and reject changed continuation evidence explicitly.
`repo.load_state_ref(saved_node, reuse_live="matching")` is the exact-state load
operation. It follows records across the connected Stores, not the current route
map. `matching`, `greedy`, and `never` are live-reuse policies; structural
similarity cannot substitute for the exact ObjectId and binding requirements.

Live construction admits every selected materializing class before it acquires a
declaration claim, restores an exact state, changes a live cache, allocates a
workspace, or invokes user construction hooks. An abstract current class raises
`TypeError` locally even when an older definition or StateRef exists. The
preflight includes materializing CDef, ObjectRef, and StateRef closures, including
reused exact nodes; `Ref` links and quotations remain inert data and are not
resolved. Definition, CDef, selector, and template construction remains
non-materializing and may describe an abstract class.

`Repo.reserve_state_graph(obj)` returns a process/thread-local, Store-neutral,
nonblocking reservation for the exact live graph. Use it as a context manager and
pass it only to `save_object(..., reservation=...)` or
`restore_state_ref_into(..., reservation=...)`. Restore preflights authority and
retained bindings, restores supplied instances dependency-first without candidate
search, and invalidates that graph for later framework state IO when a restore
hook fails. Recover by loading a fresh graph with `reuse_live="never"`.

Object aliases resolve through `get_alias()` to complete `ObjectRef` authority.
State aliases resolve with `resolve_state_selector()` to `StateRef` authority.
`fork_object_ref()` and `fork_state_ref()` are explicit Repo rekey operations;
the latter's `federated` option controls its dependency-copy behavior and is not a
save-routing compatibility spelling. Read-only query federation is also separate
from routed saving.

## Current-State Coordination

`Store.compare_and_set_state_alias(record, expected_state_ref_digest=...)` is the
Store-local current-state primitive. It holds one physical Store writer fence while
it reads the scoped alias, compares the expected digest (`None` means absent),
checks that the replacement `StateRef` is complete same-Store authority, and
replaces the alias. A mismatch raises `StoreAliasConflictError` without changing
the alias. It does not introduce another alias record format or a cross-Store
transaction.

`Repo.resolve_state_alias(object_ref, alias, store=...)` reads one selected Store
when `store` is supplied; otherwise it retains normal identical-replica resolution
and rejects conflicts. `Repo.get_or_declare_object_ref(cdef, store=...)` selects a
unique existing declaration or creates it through the existing declaration and
first-construction-claim fence. `Repo.save_object_if_current(obj, alias=...,
expected=..., store=...)` first publishes the immutable snapshot, then CAS-advances
the scoped state alias. These mutation APIs select one writable physical Store,
defaulting only when the connected writable physical destination is unique.

A CAS loser can leave its already published immutable snapshot as a safe orphan;
DRYML never deletes it to simulate rollback. Buffered ZipStore coordination commits
before success and reads a freshly reopened archive to reconcile errors after a
possible archive replacement. A proposed current target is idempotent success; an
unchanged predecessor requires a consumer reload/reapply; another valid target is
a conflict. Malformed alias scope or missing target remains an authority error.

## Persistent Metadata

`Repo.get_metadata()`, `set_metadata()`, and `delete_metadata()` read, replace,
and remove current whole mappings attached independently to Node object projections
and exact saved graphs. `scope="object"` / `scope="state"` selects the question
explicitly. Without it, selected state implies state scope, otherwise object scope;
stateless saved graphs therefore need explicit state scope. Replacements are
Store-local journaled atomic LWW writes, not field merges, and synchronously update
enabled indexing.
`SaveAnnotations(object=..., state=...)` applies optional current replacements
during save and captures the publication-time values in a new snapshot. `None`
means no explicit replacement; `{}` is a present empty mapping.

`Repo.get_snapshot_metadata(saved_node)` returns detached `NodeSnapshotMetadata`,
the immutable first-capture save evidence,
including UTC save time, environment, combined requirement outcome/coverage,
diagnostics, lineage facts, and captured annotation copies. Repeated saves,
copies, and later current edits do not refresh this record.
`Repo.get_lineage_metadata()` returns known creation evidence or explicit unknown;
it never infers creation from save or filesystem time.

Node metadata uses the closed `dryml-node-metadata/v1` family under
`metadata/{current_object,current_state,lineage,capture}/<shard>/<digest>.record`.
Object scopes use object-projection addresses; state/capture scopes use exact
saved-root addresses. Current replacements/deletions do not refresh captured
annotations. Pure decoded/prepared metadata cannot establish current holder,
source, availability, name or completeness evidence. Legacy snapshot-JSON content
facts are not admitted by current Node readers.

Exact Node loads freshly compare captured evidence among selected root holders
and reconcile lineage evidence before constructors or restore hooks. Conflicting
unqualified replicas require an explicit provider. Plans retain no authority
verdict; each use revalidates. A validated destination/source preflight may be
shared with exact equal captured graph metadata inside that private operation,
without repeating every LocalState/destination compatibility check. Known lineage
is installed on the live identity before later saves, including saves to another
Store. Low-level graph publication need not carry a descriptive capture; explicitly
requesting missing captured metadata still fails rather than inventing it.

Metadata inspection validates authority without payload reads, materialization,
probes, or optional framework imports. Multi-Store reads compare all target
holders and require explicit `store=` selection when facts conflict. Snapshot
directories and local-state sources returned by a ZipStore are borrowed from its
extraction and expire when the handle closes. See [Persistent Metadata](metadata.md)
for scopes, queries, capture, recovery, and sharing guidance.

Closing a ZipStore inside its retained writer/authority fence invalidates the
handle immediately, but extraction removal waits until the outermost native lock
has been released. This avoids deleting an open lock file on Windows. A deferred
removal failure is raised at fence exit; calling `close()` again retries cleanup.

## Portable Repo Definitions

`Repo.to_definition()` exports a detached `RepoDefinition` v1 configuration
snapshot. It contains supported Store descriptors and their order/default,
normalized routing rules, `config`, lease duration, and deletion-save setting.
It excludes Store contents, aliases, claims, main definitions, live Object
affinity, caches, sessions, archive buffers, and runtime resources. It does not
commit archives, save Objects, resolve symbols, or activate a session.

Routing selectors containing unordered graph-bearing values, normalized Nodes,
exact identity/state facets, or retained construction views use the closed
`node-selector` version-1 carrier. Its `graph` is one authenticated
`dryml-node-graph/v1` table with shared value blocks and an exact root view, not a
second graph model. Independent equal Nodes remain separate unordered members;
aliases between members and other branches survive reconstruction. Quotation,
receiving roles, normalized structure and fixed facets remain exact. Fresh decoded
handles preserve the graph's alias partition, not process-local handle equality.

Canonical transport placement resolves unordered ties over the complete table,
including retained construction views and occurrence joins. JSON bytes are stable
across hash seeds and round trips. This placement does not redefine semantic graph
identity or substitute for source-byte authentication/CAS. Compatible simple
selectors retain the existing closed descriptor spelling; its reference-path and
set-fingerprint checks remain in force. Unknown carrier, graph, identity and key
versions are rejected. No Store conversion or migration is performed.

`RepoDefinition.to_data()` and `to_json()` expose detached data for caller-owned
inspection or transport. `from_data()` and `from_json()` only validate and decode
the closed bounded grammar. They do not open Stores, resolve executable symbols,
construct a Repo, materialize Objects, or activate a session. The configuration
can include user-supplied `config` values; callers must treat exported data as
sensitive when those values are sensitive. Validation errors identify a field but
do not embed arbitrary supplied values.

`Repo.from_definition(definition)` is the separate live boundary. It validates
all descriptors, then opens fresh handles only for required existing DirStore or
path-backed ZipStore authority. Missing, inaccessible, malformed, wrong-type, or
incompatible storage raises `RepoDefinitionError`; reconstruction never creates
or repairs replacement storage and never installs a session Repo. Outside a
resource cache, the returned Repo owns its freshly opened handles.
`close(flush=True)` retains normal commit behavior, while `close(flush=False)`
releases owned resources without a commit. Reconstructed cleanup never closes
caller-supplied borrowed handles.

## Standalone Store Definitions And Resource Reuse

`Store.to_definition()` returns a detached mapping for one supported `DirStore`
or clean path-backed `ZipStore`: `{"kind": "dir", "path": "/absolute/path",
"query_index": "auto" | "sqlite" | "memory" | "none"}` or `{"kind": "zip",
"path": "/absolute/archive.zip"}`. It describes an existing location and opening
policy only. It has no Store contents, aliases, locks, caches, sessions, or archive
buffers, and it neither saves nor commits.

`Store.from_definition(mapping)` validates that same closed grammar and opens only
existing authority. Malformed mappings raise `RepoDefinitionError`; missing or
malformed authority raises the Store authority error. It never creates or repairs
storage. Dirty, file-like, missing, and zero-length ZipStores, custom DirStore
query settings, and custom Store types cannot be exported.

Inside `dryml.session.resource_cache()`, matching `Store.from_definition()`,
`DirStore.open_existing()`, and Repo reconstruction reuse a process-local Store
handle when its backend, physical identity, and supported opening settings match.
DirStore matching includes its filesystem identity and query policy; a different
query policy opens a distinct handle. Cache-owned reconstruction handles are closed
without commit at scope exit. A dirty cached path-backed ZipStore remains its same
live transaction on a matching request, but remains ineligible for portable export.
Without this opt-in cache, every existing-authority open returns a fresh
caller-owned handle.
The [Session](session.md) guide defines cache activation, inspection, and
thread/task lifetime; [Generic Execute](execute.md) uses that same facility only
within a worker setup scope.

The cache also matches a Repo by ordered Store keys, default ordering, routing,
nested JSON configuration, lease duration, and deletion-save setting. Equivalent
detached requests return one leased Repo; different behavior remains a different
Repo even when Store handles overlap. Before every lookup the cache rechecks live
Repo snapshots, so direct configuration, routing, default-order, or nested
configuration mutation cannot satisfy a stale key. A changed entry that collides
with a canonical entry remains alive for current holders but is ineligible for
new matches. Adding a physical Store to a cache-leased Repo is rejected before
opening or adopting it; reordering existing handles and changing configuration or
routing remain allowed.

Live reconstruction rejects descriptors that name the same physical Store through
duplicate, dot-segment, or symlink paths before opening duplicate handles.
Portable Selector map keys retain their `str` or `int` type, so the keys `1` and
`"1"` remain distinct after a definition round trip. If a reconstruction fails
and a newly opened handle also cannot close, it raises
`RepoReconstructionError`, chained from the original reconstruction failure.
It privately retains only fresh failed-close handles, exposes bounded immutable
`cleanup_issues`, and its serialized `cleanup()` retries every retained handle
once. Successful handles are removed; if ordinary failures remain, `cleanup()`
re-raises that same error with updated issues, and after all closes succeed it is
a no-op. Direct `BaseException` control flow, including `KeyboardInterrupt` or
`SystemExit`, continues to propagate with the owner error attached as
`repo_cleanup_error`, preserving retry ownership. It never retains or closes
caller-supplied borrowed handles.

A Repo configured with `save_objs_on_deletion=True` attempts a detached
strong-cache snapshot in cache order when it is collected. The first save failure
stops later deletion saves, but owned resources still receive non-flushing close
attempts. Cleanup failure is reported through Python's unraisable-hook mechanism
with a fixed bounded message; it does not include the original exception, its
traceback, or the Repo representation. Explicit `close(flush=False)` never runs
deletion saves.

Definition export rejects dirty, missing, or zero-length ZipStore archives,
file-like archives, unsupported Store types/settings, nonportable configuration,
ambiguous built-in physical destinations, and explicit custom clocks or owner
token factories. A definition describes configuration, not a frozen data snapshot:
use StateRefs to request exact saved Object state.

### Core Execute Shared Storage

`dryml.core.execute` is a separate, same-host execution adapter, not a general
Repo transport. Its initial `SharedDirStoreStrategy` accepts only an existing
direct `DirStore` table that a worker can reopen at the same physical locations.
At submission it exports the selected live Repo once, freezes that one
`RepoDefinition` and Store order, and opens one fresh coordinator-owned recovery
Repo. A separate direct control Store is represented explicitly; a Store already
in the table is represented by its table index and is reopened once in the worker.

The adapter rejects missing storage, ZipStore authority, inaccessible or ambiguous
physical destinations, unsupported query policy, and non-direct strategies before
worker submission. It neither creates a Store nor substitutes a replica. The
definition is configuration only, so input Objects are not auto-saved. Worker
publication uses normal Repo save authority; result/update evidence contains only
exact StateRefs and Store-table-relative facts. Recovery and worker teardown close
their freshly opened handles with `flush=False` and never close the caller's Repo
or Store. See [Generic Execute](execute.md) for the execution lifecycle.

A managed Artifact worker publishes its final state through the same ordinary
Repo/managed authority. Its coordinator-visible result is the exact final
`StateRef`, and a later reader reconnects only the required state Store to restore
the result. The worker does not transfer live source/model payloads or managed
control records to that reader; those references remain required only for a later
compute or rerun.

### CachedDataset Working Storage

CachedDataset allocates unpublished work only beneath one routing-selected direct
DirStore's private local-state staging namespace. Checkpoints persist an opaque
token and owner/operation/attempt evidence, never a path or directory contents.
Lookup probes only already connected Stores and requires exactly one matching
validated child; it does not create storage, follow current routing to another
destination, or use inode/device identity. Retained work survives ordinary
process exit. Cleanup separately requires explicit reclaimable evidence and the
work lease; owner loss alone never authorizes deletion.

This working dependency is distinct from final publication. A completed cache
copies its sealed chunks into the immutable snapshot selected by ordinary routing,
replication, or one explicit `store=` closure override. Removing completed work
therefore does not affect the final StateRef, while removing progress work makes
that checkpoint non-resumable. Only direct local DirStore work and same-host
access are qualified; ZipStore work allocation, distributed filesystems, remote
workers, and automatic handle refresh are not implied. A caller that observes
writes from another process must use the Repo/Store's existing explicit refresh
or reopen behavior.

## Store Authority And Lifetime

`DirStore` is the supported directory checkpoint backend. Its immutable
definitions, complete v3 snapshot directories, snapshot-local states, lineage,
declarations, claims, aliases, main references, and mutable current metadata are
authoritative. SQLite query indexes, metadata projections, record-reference
indexes, caches, and dirty markers are derived state and can be rebuilt without
replacing Store records. New and old incompatible Store formats fail before
hydration or index activation; Store v2 is not rewritten and has no fallback
reader.

Repo borrows supplied Store instances. It owns and closes Store handles it opens
from paths or file-like inputs, including reconstruction handles. `add_store()`
and `set_default_store()` may open a specification but do not move stored state.
Closing a borrowed Store while its Repo uses it is unsupported. See
[Formats](formats.md) for durable record layouts and [Managed Operations](managed_operations.md)
for lifecycle control and Store/ObjectId ownership.
