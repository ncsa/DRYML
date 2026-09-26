# Repos and Stores

`Repo` coordinates live Object realizations, exact StateRef persistence, and one
or more connected Stores. A Repo is local process authority, not a transport or
security boundary. Store data, definitions, symbols, and serialized payloads are
trusted DRYML inputs; supported shared use is same-host use on a local filesystem
with the documented locking, atomic-replace, and SQLite semantics. Distributed
filesystems and cross-host coordination are unsupported.

## Saving With Routing

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
`Repo.load_state_ref(state_ref, reuse_live="matching")` is the exact-state load
operation. It follows records across the connected Stores, not the current route
map. `matching`, `greedy`, and `never` are live-reuse policies; structural
similarity cannot substitute for the exact ObjectId and binding requirements.

Live construction admits every selected materializing class before it acquires a
declaration claim, restores an exact state, changes a live cache, allocates a
workspace, or invokes user construction hooks. An abstract current class raises
`TypeError` locally even when an older definition or StateRef exists. The
preflight includes materializing CDef, ObjectRef, and StateRef closures, including
reused exact nodes; `Ref` links and quotations remain inert data and are not
resolved. Definition, CDef, selector, and search-space construction remains
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

## Persistent Metadata

`Repo.get_metadata()`, `set_metadata()`, and `delete_metadata()` read, replace,
and remove current whole mappings attached independently to exact ObjectRefs and
StateRefs. Replacements are Store-local atomic LWW writes, not field merges.
`SaveAnnotations(object=..., state=...)` applies optional current replacements
during save and captures the publication-time values in a new snapshot. `None`
means no explicit replacement; `{}` is a present empty mapping.

`Repo.get_snapshot_metadata()` returns immutable write-once save evidence,
including UTC save time, environment, combined requirement outcome/coverage,
diagnostics, lineage facts, and captured annotation copies. Repeated saves,
copies, and later current edits do not refresh this record.
`Repo.get_lineage_metadata()` returns known creation evidence or explicit unknown;
it never infers creation from save or filesystem time.

Metadata inspection validates authority without payload reads, materialization,
probes, or optional framework imports. Multi-Store reads compare all target
holders and require explicit `store=` selection when facts conflict. Snapshot
directories and local-state sources returned by a ZipStore are borrowed from its
extraction and expire when the handle closes. See [Persistent Metadata](metadata.md)
for scopes, queries, capture, recovery, and sharing guidance.

## Portable Repo Definitions

`Repo.to_definition()` exports a detached `RepoDefinition` v1 configuration
snapshot. It contains supported Store descriptors and their order/default,
normalized routing rules, `config`, lease duration, and deletion-save setting.
It excludes Store contents, aliases, claims, main definitions, live Object
affinity, caches, sessions, archive buffers, and runtime resources. It does not
commit archives, save Objects, resolve symbols, or activate a session.

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
