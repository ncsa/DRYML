# Managed Operations

Managed member arguments use the shared [signature contract](signatures.md),
including explicit Node metadata refinements and materializing Object roles.

`dryml.managed` owns the lifecycle of state-mutating Object methods. Its
durable-result lifecycle supports explicit checkpoints, optional immutable initial state,
exact final selection, detached completion replay, deterministic starts,
two-phase rerun, resume, operation-scoped ownership, fenced control CAS, metadata
history, and independent status/interruption control. Importing Core does not
import managed, Execute, Dispatch, or optional frameworks.

## Integration status

The previous StateRef/current-v2 execution path is retired. A decorated direct call
now validates ordinary arguments once, selects admission, prepares/restores a
selected saved start and executes the authored body with `ManagedContext`.
Normal success returns `InvocationResult`: `.value` is the durable ordinary return,
`.final_state_ref` is historical SavedNode metadata, `.status` is detached status,
and `.execution` distinguishes operation, run, current caller and producer IDs.
Compatible completed calls replay this metadata without receiver payload IO or
mutation, even on an older compatible live receiver. A pending-repair
call returns an explicit `AdmissionOutcome`; that is reconciliation, not ordinary
method success. Declarations retain their checked synchronous signature rules.

Core Execute's managed-owned adapter admits detached targets before materialization;
it shares direct-call start, replay and recovery policy. Models and Artifacts consume
the same InvocationResult and SavedNode contracts. Bounded local/subprocess consumer
proofs and optional-backend limits are recorded in
[Object State Lifecycle Verification](object_state_lifecycle_verification.md).
Acquiring low-level admission or constructing metadata alone does not retain
private state. U15's selected saved-start handshake does acquire physical grants.

### Invocation-local performance and evidence

Snapshot environment evidence is observed lazily before the first capture's state
reservations, once per executing invocation. Later checkpoints and automatic final
capture reuse that observation, with independent capture timestamps and requirement
collection. An unavailable observation remains unavailable for that invocation;
resume/rerun starts a fresh observation. Replay and explicitly selected finals that
need no capture perform no environment inventory.

Admission also owns a caller-local, bounded memo of immutable metadata
interpretations. Exact source bytes, reader implementations and interpretation
parameters distinguish entries; encoding facts are attached to immutable Node
handles. At most 64 entries and 4 MiB of encoded source/output bytes are retained,
with interpreted values additionally bounded by their closed codec grammars. The
outer admission clears every entry on normal return, failure or cancellation.
Inherited process/thread/task contexts cannot use another caller's memo.

This memo shares decoding, canonical JSON validation, and pure Node interpretation
across control and Store readers. It does **not** cache file reads, pending flags,
authority cuts, ownership, generations, payload availability, grants or publication
acknowledgements. Every safe point rereads current bytes; every selected-state
reader and external override is still freshly consulted. Changed/malformed bytes
cannot match an old interpretation, and a hash alone is never a cache/equality
proof. See [performance measurements](managed_lifecycle_performance.md).

### Execute transport boundaries

[Generic Execute](execute.md) supplies detached invocation-graph transport;
the managed-owned adapter interprets managed declarations, configuration and
completion results. Generic Execute does not inspect status or
rerun a managed method automatically. Admission, selected-start restoration and
completion replay remain managed policy, including when invoked in a worker.

Explicit transported configurations carry supported direct-DirStore descriptors
and frozen callback membership, rather than live Repo/Store resources. Workers
reopen those descriptors through their Session resource cache; unsupported
resource types have no transport fallback. See [Repos and Stores](repos.md) for
authority and resource lifetimes. Trusted same-host transport does not provision
environments or promise automatic code migration in long-lived sessions; the
[code-shift limitation](solutions/architecture-patterns/2026-09-18-code-shift-in-long-lived-sessions.md)
continues to apply.

## Caller configuration and start selection

`ManagedConfig` remains immutable and keyword-only. It accepts borrowed
`state_repo`/`control_store`, exact-boolean `rerun`, optional `start_from` SavedNode,
and the caller-owned checkpoint callback list. `snapshot()` retains the immutable
start metadata and copies callback membership into a tuple. Config construction
checks saved completeness only; it does not promise current availability.

```python
config = ManagedConfig(
    state_repo=repo,
    control_store=control_store,
    rerun=True,
    start_from=checkpoint_b,
)
result = experiment.train(epochs=20, managed=config)  # user-authored managed method
value = result.value
final = result.final_state_ref
```

Supplying `start_from` without `rerun=True` fails with `start_requires_rerun`
**before restore or cleanup**, on absent, completed and unfinished operations alike.
Rerun selects supplied start, otherwise the operation's immutable recorded initial,
otherwise current live state. It never opens an unselected initial payload and
never falls back from an unavailable/incompatible selected start. A saved start
seeds latest recovery without reserialization; a live start creates no checkpoint.
An override changes neither operation identity nor the recorded initial default.

Ordinary compatible completed calls select replay. Ordinary unfinished calls
require both `resumable=True` and a usable latest recovery checkpoint, restore
that checkpoint, retain the run ID, and allocate a fresh invocation ID. Changed
normalized ordinary arguments require explicit rerun. Missing latest recovery
does not fall back to initial or live state.

Rerun is two phase: journal a proposed run/start token while old current remains;
validate selected state, prepare/protect physical grants, consume Core's one-shot
`Repo.prepare_restore`, confirm the saved recovery association, then admit by exact
owner/generation CAS. Only after confirmed association/admission are superseded
non-durable start/latest/final roots released. Initial, durable and independent
user-save grants survive. Garbage reclamation uses full StateKeys after all
source leases end. Only already-unreferenced physical garbage-removal OSErrors
become a cleanup report; grant, publication, control and index failures propagate.

A no-selected-start live rerun may still borrow deferred bytes from its former final
(for example a completed CachedDataset). Its old final association is retired at
admission, with the grant carried as explicit `release_pending` debt until the new
final publication replaces it. This does not select or restore that old checkpoint,
designate an initial state, or fabricate latest recovery. Failed execution retires
the debt; a failed retirement remains inspectable on the primary exception as
`cleanup_error`/`cleanup_outcomes`. A subsequent invocation reconciles that debt
without execution before a separate retry. Required grant/index failures are never
converted to successful physical-garbage reports.

Before-hook validation failure leaves the live receiver usable. Hook failure can
invalidate it while preserving old persisted current; there is no rollback.
Uncertain CAS retains exact previous/proposed fingerprints and both resource sides.
Accepted interruption advances the independent generation fence and is checked
at admission CAS and the execution handoff.

Pending recovery **terminates**: unadmitted preparation is fenced/abandoned and its
exact resources reconciled, while old current survives; admitted preparation keeps
its run/latest; incomplete finalization becomes a failure preserving latest;
recorded completion is inspected without final payload reads. No receiver restore,
capture, callback or body is replayed. A separate invocation must freshly prepare
or restore for later work. There is no cross-invocation restore receipt. Complete
result/final evidence is confirmed idempotently and completed without normalizing
again or replaying consumer hooks. A prepared final association can be completed
after a crash only when its token, run, invocation and generation match the
preparation and its exact saved selection, resources, adopted physical grants,
result and publication acknowledgements are freshly confirmed. A former run's
`release_pending` final is not a candidate. An incomplete capture records failure and
preserves latest recovery; a separate resumable invocation must restore freshly.

## Author checkpoint and selection controls

The injected context exposes read-only `initial_state_ref`, `latest_checkpoint`,
`final_state_ref`, `is_resuming`, operation/run/invocation IDs and generation.
`cleanup_report` exposes accumulated physical-garbage-only removal outcomes.
References are immutable SavedNode metadata; they are not cached availability
proofs. Context mutation is limited to its original active process/thread/task.
Retired or foreign use raises `inactive_context`, recursive callback mutation
raises `recursive_checkpoint`, and stale generation/ownership fails before capture.

```python
@managed_operation(resumable=True)
def train(self, epochs, *, managed):
    if managed.initial_state_ref is None:
        managed.checkpoint(initial=True)
    for epoch in range(epochs):
        self.train_epoch()
        best = managed.checkpoint(durable=True)
    managed.final_state(best)
    return {"epochs": epochs}
```

The body above returns an ordinary dictionary. Its caller receives InvocationResult
after the dictionary, exact selected final, required grants and index prerequisites
are confirmed and completion's own fenced mutation is acknowledged.

`checkpoint(states=None, durable=False, initial=False)` uses Core's routed save
batches. `states` accepts a partial StateSpec or rooted LocalState map: explicitly
selected compatible states skip capture, and each remaining distinct live stateful
node captures once. The returned SavedNode is complete, including for an empty or
entirely stateless graph. No checkpoint is implicitly initial. `initial=True`
designates it explicitly; `initial_state(saved)` can instead select an existing
complete graph, including the first designation during resumed execution. Exact
repeat designation is idempotent; another initial raises `initial_conflict`.
Equality preserves facets, sharing, roles, quotations **and retained construction
evidence**, and does not use packet placement or hashes as sufficient proof.

`final_state(saved)` checks the receiver projection against the original supplied
graph, current live codec compatibility and fresh state availability, then protects
that unchanged selection with an independent final root. It neither saves nor
restores the live receiver. Earlier or mixed-point states are allowed when
compatible; the live object may remain later than final. Selecting a final protects
it even if a subsequent checkpoint supersedes latest. Designation does not record
completion, and a later exception or accepted interruption still prevents success.

Each explicit checkpoint associates complete latest recovery and required grants/
index acknowledgements **before** running its configured callbacks in list order.
A callback exception reaches the author, stops later callbacks in that event and
leaves the associated checkpoint latest. An uncaught exception fails the invocation;
a caught ordinary checkpoint/callback error can continue to valid state finalization.
Confirmed uncommitted association preparations settle without hooks before later
work. Uncertain control/index work remains explicit recovery, not a success guess.
Callbacks run outside capture reservations, Store writer and short control locks.

The result owner executes the authored body under the same owner,
then normalizes/encodes the ordinary value **before** requesting state
finalization. It uses the explicit final unchanged, or captures once if no final
was selected. Automatic final capture invokes zero checkpoint callbacks. An explicit
terminal checkpoint selected as final therefore has one capture/one callback, with
no second capture or callback at finalization. Owning return normalization runs
once; separately nested `@function` calls keep their own boundaries. Replay runs
ordinary argument normalization for compatibility but no return normalization.

## Durable ordinary values and fresh lazy access

The result grammar is closed:

- Exact Python `None`, `bool`, `int`, `float`, `complex`, `str`, `bytes`.
- Exact `list`, `tuple`, `dict`, `set`, `frozenset` recursively containing supported
  values. Keys/set members must be admitted hashable values.
- Nonobject NumPy scalars and exact `ndarray`, using the common explicit dtype
  codec, including structured nonobject dtypes, shape, byte order and logical bytes.
- Explicit immutable Core metadata: Definition/ConcreteDefinition/ConcreteNode,
  ObjectId, StateKey, LocalState, IdentitySpec and StateSpec. Graph codecs preserve
  sharing, quotation and retained construction and never materialize metadata.
  Every metadata root in one result uses one shared Node table and one decode
  owner. Returning a parent alongside its child, or two parents sharing a child,
  preserves that same occurrence across values; independent recipe-equal nodes
  remain independent. IdentitySpec/StateSpec roots use the same table too.

Containers retain distinct tags, dictionary insertion order, DAG aliases and scalar
floating bits (including negative zero, infinities and NaN payloads). Cycles,
arbitrary subclasses, custom/opaque values (including frozen summary dataclasses),
callables, generators, live Objects/resources and object-dtype arrays are rejected.
Return a supported container for a custom summary. There is no pickle/dill or
resource fallback. The separate managed result codec defaults to an 8 MiB byte
budget, 65,536 nodes/container entries, depth 80 and 262,144 integer bits;
the node budget includes shared metadata handles and their internal wrappers,
and metadata graph depth includes retained construction edges. Malformed shared
views, tags, references, versions and bounds raise `ManagedResultError`.
Unsupported returns fail before automatic final capture.

Result bytes are immutable and content addressed at
`managed/results/v2/<shard>/<sha256>/current.json` in the selected control Store.
Their result kind/version/key is retained in current and history metadata. Result
metadata embeds its complete metadata closure; references in a value **do not**
grant retention of private StateKey data. State cleanup therefore does not prevent
status, replay or value access while result bytes remain.

The closed `dryml-managed-result` packet and typed result reference are version 2.
The packet has exactly `schema`, `version`, `nodes`, `root`, and `metadata` fields.
`metadata` is null when absent, otherwise one `dryml-node-graph/v1` identity-3 block
with a single shared table and exact authenticated root views. Each `metadata`
result row contains a view index, rather than a separate graph blob. All views
must be referenced by reachable result rows; forged views, unused blocks/roots,
and invalid wrapper values fail closed. Managed Execute adapter version 2 carries
only the typed v2 reference and detached source descriptor. Result packet/reference
v1, adapter v1 and the old result storage namespace are incompatible and are
rejected with original bytes intact, without migration or resource fallback.

InvocationResult retains only detached values and a `ResultSource` (canonical local
DirStore path plus persistent UUID), never an open Repo/Store, connection, reader,
lease or availability proof. Closing the original Repo or worker leaves status,
IDs and reference inspection usable. Every `.value` access opens a **fresh** source,
validates its UUID and result kind/version/key, protects the byte-read lifetime,
checks the content hash and decodes anew. Missing/corrupt result bytes, an absent
source or another Store installed at the same path raises `ManagedResultError`
only on lazy access, with no reexecution. A prior decoded value cannot extend a
stale source's validity. Unsupported source kinds have no serialization substitute.

Completion replay preserves the original run and producer invocation, while
allocating a fresh caller invocation. It runs no body, restore, capture, checkpoint
callback, owning return normalization or `_managed_post_publication` consumer hook.
That optional private consumer receives `(final_saved_node, borrowed_repo, report)`
only after a new completion is acknowledged; `report` is None for selected finals.
Its failure propagates while completion remains recorded and is never redelivered.
The raised consumer error exposes detached `.invocation_result` metadata rather
than a failed lifecycle projection.

Uncaught authored/result errors retain execution IDs, optional
`checkpoint_state_ref`, and detached `.failed_result` inspection. Publication or
index ambiguity retains its authoritative mutation/cut evidence instead of
pretending to contain an ordinary successful value. A committed completion whose
own index projection fails remains completed, raises the explicit partial error,
and withholds ordinary delivery. A later repair call terminates with
`AdmissionOutcome("completion_recorded", ...)`; a subsequent call replays the value.

Managed publication uses provisional internal Core grants, not a permanent user-save
root for each checkpoint. Root association reuses the two-phase preparation journal:
prepare/protect new physical grants, confirm adoption, CAS current, acknowledge the
owning control mutation, then retire only superseded non-durable start/latest/final
tokens. Initial, durable, independent user and active-reader protection are separate.
Partial Core publications preserve their exact provisional resource ledger/report;
they never become latest just because bytes exist. Pending recovery performs no
capture, restore, body or callbacks. Later resumable execution freshly restores.

### Explicit Retention And Cleanup By Operation

During a body, `managed.retain(saved)` adds an independent explicit root for a
complete compatible SavedNode, without capture, restore or private-state serialization.
Repeating the exact retained selection is idempotent. `managed.release(saved)`
releases matching explicit and durable roots only: initial/latest/final roots,
another operation, independent user saves and active readers remain effective.
Both validate the original active caller and owner/generation fence. Released root
metadata remains inspectable. Explicit roots and durable checkpoints survive reruns
until explicitly released.

```python
first = managed.checkpoint()
managed.retain(first)
# Later, after another checkpoint replaces latest:
outcomes = managed.release(first)
```

Outside a body, use the selected control authority and detached operation ID:

```python
from dryml.managed.control import ManagedControlStore

control = ManagedControlStore(control_store, repo)
roots = control.list_checkpoints(result.execution.operation_id)
outcomes = control.cleanup_checkpoints(
    result.execution.operation_id, tokens=[root.token for root in roots]
)
historical_roots = control.list_checkpoints(
    result.execution.operation_id, include_released=True
)
```

Each root exposes its kind, token, saved selection, disposition and exact affected
Store/full-key resources. Listing reads bounded current metadata, not payloads or
all operation history. `tokens=None` explicitly cleans all remaining associations,
including initial/latest/final/durable roots. Unknown tokens fail before release.
Cleanup takes nonblocking operation ownership: an active invocation returns an
`already_running` conflict immediately rather than waiting or interrupting it.
Unsettled preparation requires separate reconciliation before explicit cleanup.

Cleanup journals `release_pending` before releasing grants, then acknowledges
`released` without erasing historical checkpoint/final/result metadata. A disconnected
or failed grant target stays pending/protected. Reconnect its exact Store and retry
the same tokens; acknowledged targets are not granted authority over other roots.
The returned per-target `StateCleanupOutcome` entries use the Repo cleanup contract
in [Repos and Stores](repos.md#inventory-explicit-retention-and-federated-cleanup).
Independent targets continue after ordinary target failure; unsafe dependent work
is unattempted. Grant/association/withdrawal/index failures never become successful
garbage reports. Required failures in automatic checkpoint/rerun cleanup propagate
with `.cleanup_outcomes`; independent completed deletions are preserved.

An active foreign reader can leave a released root's payload `in-use`; no deletion
is queued, so retry `repo.delete_states([key])` after that reader exits. Release
inside the same caller's key scope instead reports `in-use` without retiring its
root; retry release after leaving the scope. All own reader/publication leases end
before automatic cleanup enters a separate exclusive deletion phase, never an upgrade.

Only failure to remove already-unreferenced private garbage after confirmed
authority/index changes preserves checkpoint/result success. `managed.cleanup_report`
and `InvocationResult.cleanup_report` expose `(domain, key, removal_outcome)` entries;
retry the exact incarnation via `store.retry_state_removal(key, removal_outcome.incarnation)`.
Ordinary `InvocationResult` success is unchanged. Historical completion, exact saved
references and `.value` remain inspectable after state cleanup. Loading reclaimed
state raises unavailable; a compatible completed call replays its result without
body, capture or restore. A separately unavailable ordinary-result source fails only
on `.value`, without reexecution. No per-checkpoint whole-Store liveness scan or
deletion of history is used to repair derived indexes.

`interrupt(cause=None)` validates the optional BaseException cause, makes one explicit
checkpoint, then requests and records interruption; the resulting ManagedInterrupted
chains that cause. External requests need not capture unsaved live state. Safe points
check fresh run/invocation/generation/request authority before state association and
after callbacks/body return. Catching an Exception cannot turn an accepted request
into success, including after final selection. KeyboardInterrupt preserves latest
without an extra capture and is surfaced as ManagedInterrupted with its cause.

## Identity and metadata-first admission

The selected control authority domain and stable **operation ID** identify a
logical operation. The operation digest uses the receiver's canonical object
projection (`dryml-object-projection/v2`) plus its declared managed member:

- ObjectIds, sharing, topology and typed link roles are retained.
- State selections on graph references are omitted recursively.
- Quoted recipe/configuration data stays exact, including checkpoint pins inside
  those quotations.
- Normalized ordinary arguments are **separate compatibility evidence**. Changing
  arguments does not fork operation identity; ordinary reuse/resume requires a
  match. Higher-level explicit rerun preparation may select new arguments.

Three identifiers have deliberately different meanings:

| Identifier | Lifetime |
|---|---|
| Operation ID | Stable projection-plus-member digest within selected control authority |
| Run ID | UUID allocated on first start or explicit new run; retained across resume |
| Invocation ID | Fresh UUID for every admission, including a completed-replay caller |

`producer_invocation_id` records which invocation produced completion. A later
caller has its own fresh invocation ID and does not replace that producer merely
by inspecting historical completion. `attempt_id` and `expected_attempt_id` are
removed in this breaking cutover; use run/invocation/generation preconditions.

`AdmissionIdentity.from_call(receiver, declaration, args, kwargs)` accepts a
detached concrete Node and a checked managed declaration. Binding/default
normalization never needs a live Object. Stateful receiver identities must
already be bound. Ordinary values use the bounded closed argument grammar;
unsupported values fail without custom serialization hooks. A signature adapter
with explicitly selected authority can construct `AdmissionIdentity` from its
normalized argument digest directly.

`ManagedControlStore.admission(identity)` acquires ownership and examines selected
control metadata before any realization. Its `OperationAdmission` exposes
`current`, stable `run_id`, fresh `invocation_id`, and an execution-only
`realize(callable)` seam. That callable receives the detached Node while the
**same owner remains held**. Transport integration must use this seam before
receiver reconstruction, so completed replay selection need not open a reclaimed
receiver checkpoint. Recovery admissions report `reconciled=True` and refuse
realization: reconciliation terminates separately from execution.

The managed-owned internal `runtime.admit_execution(identity, declaration,
config.snapshot(), control, realize=..., execute=...)` layers U15 policy on that
same admission. It accepts detached receiver and normalized argument authority;
only an execution branch calls `realize` with state-unpinned receiver metadata,
then performs the selected one-shot restore. The same lifetime owner remains held
through `execute(admission, running_snapshot, live, is_resuming)`. This integration
callback is not a public completed-operation result or a replacement state machine.

## Operation ownership and shared mutation

Ownership is **one nonblocking lifetime operation lock**, keyed by selected
control authority domain plus operation ID. A duplicate same-operation admission
raises `ManagedConflictError("already_running", ...)`; it does not wait, join or
take over. Different selected control Stores are independent authority domains.
Callers must choose the intended domain consistently.

There is no persisted Cartesian descendant Store/ObjectId exclusion and no
process-local graph-wide managed overlap rejection. Distinct operations can share
saved dataset identities and the same read-only live dataset, including across
threads, tasks and same-host spawned workers. Each operation has independent
control/results/provenance ownership. **Callers synchronize shared live mutation**,
including reads/captures/restores affected by that mutation. Core's exclusive
state-IO reservations still protect actual overlapping capture/restore/reuse;
they are not an operation-lifetime lock.

An ownership token belongs to its creating process, thread and asyncio task.
Inherited, foreign and retired tokens cannot mutate authority or release another
token's lock. Active invocation ID and generation are mandatory owner fences for
checkpoint, interruption, completion and resource-release authorization. Root
intent metadata retains exact affected resources and irreversible dispositions;
stale callers cannot modify another invocation's ledger. A metadata authorization
is not a transferable physical grant-release receipt.

## Independent short control

The lock order for paths that need multiple resources is:

1. Lifetime operation owner, for an admitted execution owner.
2. StateKey leases globally ordered by Store authority domain and full StateKey.
3. Store writer fences ordered by stable authority domain.
4. Short SQL transactions **or** short control CAS, without overlapping them.

Short control never acquires/waits for an earlier-order lock, performs graph
preparation, runs a hook, or reads private state. Builders and metadata decoding
run outside short control. Store journal registration/invalidation precedes CAS;
classification, Store projection/SQL maintenance and acknowledgement occur after
control is released. Pending journal evidence invalidates coverage through that
gap. Projection failure preserves committed control and mutation/cut evidence;
it is an actionable partial outcome, not permission to overwrite completion.

Control publication never waits for short control while retaining a Store writer
fence. Its registered mutation remains protected through a nonblocking short-lock
acquisition; contention refuses with `control_busy` and retained cancelled-cut
evidence for reconciliation after Store fences end. This avoids a gap in which an
independent Store recovery could acknowledge an uncommitted cut before its CAS.

`inspect(operation_id)` reads current detached metadata. `status(operation_id)`
additionally observes ownership without taking or waiting for the lifetime lock.
Owner loss requires a successful nonblocking operation-lock probe **before** the
final short control read, with unchanged run/invocation/generation. Timeouts and
descendant locks never prove death. Status projects proven loss as
`failed`/`owner_lost` without rewriting current authority.

External interruption accepts optional `expected_run_id`,
`expected_invocation_id`, and `expected_generation`. It reports stale preconditions
without mutation, or commits a request and advances generation. Accepted
interruption fences the old owner's checkpoint/success boundary. These controls
remain responsive while an admitted owner holds its lifetime lock; they do not
guarantee that another safe point or checkpoint will occur.

## Persistent authority and historical inspection

The selected control DirStore owns:

- Area envelope: `dryml-managed`, **format 2**.
- Current authority: `dryml-managed-current`, **record version 4**.
- Pending authority: `dryml-managed-pending`, **record version 2**.
- Version-1 named managed Node records and version-2 history with result references.
- Version-3 root intents with explicit kind, SavedNode selection and provisional
  publication token/exact Store-ack evidence; earlier versions are rejected.
- Version-1 Store acknowledgement evidence (UUID, exact mutation UUID and its own
  journal epoch, plus optional provisional token), not a cached availability proof.
  Finalizing current retains the typed result reference and its result publication
  acknowledgement before automatic capture; associated final intents retain capture
  and provisional evidence so recovery can finish token retirement without recapture.
- Version-2 result packets/references and the `managed/results/v2` namespace;
  managed Execute adapter version 2 validates nested result references before
  reconstruction or resource opening. Current record version 4 and history version 2
  retain these explicitly versioned nested references.
- Version-1 start/root-association preparation sidecars under
  `managed/operations/v2/starts/v1/<shard>/<operation>/current.json`, containing exact
  previous/proposed current metadata, the physical retention token, disposition
  and reconciliation acknowledgement. These are intents, not current authority
  or serialized process-local restoration capabilities. The same bounded sidecar
  now also journals checkpoint/initial/final root replacements, with no second
  operation-current state machine.
- Nested `dryml-node-graph/v1` / identity-3 graph, reference and StateKey codecs.

Area format 2 does **not** admit old current-record versions 2 or 3. Every owning reader
rejects incompatible nested versions as well as old area/current/pending records,
preserving source bytes. No old authority is automatically converted, deleted or
overwritten. A pending replacement is reconciled only against its exact
previous/proposed generation and byte fingerprints under operation ownership.

Decoded current and start-preparation records retain their validated source bytes
as a private bounded immutable fact. This preserves exact CAS evidence when fresh
Node handles reorder unordered containers or authenticated packet tables. It is
not cached external proof: reads and publication still freshly validate persisted
bytes, ownership and generation. Replacing a record discards its source-byte fact;
new values use canonical JSON with valid Node packets. Nested previous/proposed
snapshots preserve their original packet evidence. Logical snapshot, history and
root-selection equality uses exact Node graph comparison (including facets,
aliases, roles and quotations), never packet placement or digest equality alone.
Logically equal graphs need not have identical storage bytes.
Advancing preparation from prepared to ready validates unchanged non-intent fields
logically when root intents change; it never manufactures a source-byte witness by
re-encoding a replacement. The sidecar and previous-current CAS still compare
their exact expected bytes against freshly read authority.

Current/history snapshots retain bounded detached receiver and SavedNode metadata.
Status and replay admission do not open any referenced private payload, even after
checkpoint reclamation. Metadata existence alone neither protects state from
deletion nor proves availability. Actual execution/restoration must establish
those separate contracts at its own admission boundary.

The managed-owned projection adapter journals current, history, preparation,
root/finalization associations and ordinary-result records and contributes typed
`lifecycle` rows alongside Core's generic `record` projection. Core never imports
the managed owner. Coverage contract 3 certifies the implemented families only when
the complete owner reader is registered; a cold rebuild with managed authority
requires that adapter. Pending invalidation blocks refresh-disabled reads, including
negative/count consumers. Completion journals its **own** invalidation before CAS,
releases control before SQL/coverage commit, and acknowledges that exact mutation
afterward. Earlier final-state acknowledgements cannot cover completion. Repair is
metadata-only and performs zero body/capture/restore/return/consumer-hook work.

## Worker diagnostics

The accepted same-host Execute/Core worker diagnostic behavior is unchanged:
bounded **unredacted raw tracebacks** are the default for worker failures and can
contain exception text, source lines, local paths, private inputs or credentials.
Use `raw_traceback=False` on the selected backend for secret-bearing work. This
accepted diagnostic exception does not change other metadata/logging protection,
make dispatched code untrusted, or imply a security sandbox.
