# Managed Operations

`dryml.managed` owns the lifecycle of state-mutating Object methods. Its **U16
author lifecycle** supports explicit checkpoints, optional immutable initial state,
exact final selection, detached replay selection, deterministic starts,
two-phase rerun, resume, operation-scoped ownership, fenced control CAS, metadata
history, and independent status/interruption control. Importing Core does not
import managed, Execute, Dispatch, or optional frameworks.

## Integration status

The previous StateRef/current-v2 execution path is retired. A decorated direct call
now validates ordinary arguments once, selects admission, prepares/restores a
selected saved start and executes the authored body with `ManagedContext`.
**Ordinary result delivery is not integrated yet.** A normal body return raises
`ManagedControlError("result_policy_not_integrated", ...)` before automatic final
capture and records failure with latest recovery preserved. Compatible completed
calls select replay without
receiver payload IO, then raise `result_policy_not_integrated` with detached
`.admission` metadata until durable-result delivery is integrated. A pending-repair
call returns an explicit `AdmissionOutcome`; that is reconciliation, not ordinary
method success. Declarations retain their checked synchronous signature rules.

This is not the completed managed runtime. U17 owns ordinary return normalization,
durable result encoding, completion and replay delivery; U18
owns model/artifact caller integration; U19 integrates the detached target into
Core Execute transport. The new `start_from` policy is currently local admission
policy, not a qualified model-wrapper or Core Execute transport capability.
Acquiring low-level admission or constructing metadata alone does not retain
private state. U15's selected saved-start handshake does acquire physical grants.

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
# Authored execution/checkpoints work; normal result delivery still raises the
# U17 integration error described above.
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
result/final evidence acknowledgement belongs to U17's future result format.

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

The body above can publish checkpoints and select final state now. Its normal
return still raises the documented result-integration error; it does not currently
deliver an InvocationResult or claim completed authority.

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

The private result-owner handoff executes the authored body under the same owner,
then lets U17 normalize/encode the ordinary value **before** requesting state
finalization. It uses the explicit final unchanged, or captures once if no final
was selected. Automatic final capture invokes zero checkpoint callbacks. An explicit
terminal checkpoint selected as final therefore has one capture/one callback, with
no second capture or callback at finalization. This internal seam is qualified by
focused tests; public completion/result delivery remains U17 work.

Managed publication uses provisional internal Core grants, not a permanent user-save
root for each checkpoint. Root association reuses the two-phase preparation journal:
prepare/protect new physical grants, confirm adoption, CAS current, acknowledge the
owning control mutation, then retire only superseded non-durable start/latest/final
tokens. Initial, durable, independent user and active-reader protection are separate.
Partial Core publications preserve their exact provisional resource ledger/report;
they never become latest just because bytes exist. Pending recovery performs no
capture, restore, body or callbacks. Later resumable execution freshly restores.

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
- Current authority: `dryml-managed-current`, **record version 3**.
- Pending authority: `dryml-managed-pending`, **record version 2**.
- Version-1 named managed Node and history records.
- Version-2 root intents with explicit kind and SavedNode selection evidence;
  version-1 root intents are rejected without conversion.
- Version-1 start/root-association preparation sidecars under
  `managed/operations/v2/starts/v1/<shard>/<operation>/current.json`, containing exact
  previous/proposed current metadata, the physical retention token, disposition
  and reconciliation acknowledgement. These are intents, not current authority
  or serialized process-local restoration capabilities. The same bounded sidecar
  now also journals checkpoint/initial/final root replacements, with no second
  operation-current state machine.
- Nested `dryml-node-graph/v1` / identity-3 graph, reference and StateKey codecs.

Area format 2 does **not** admit old current-record version 2. Every owning reader
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

Current/history snapshots retain bounded detached receiver and SavedNode metadata.
Status and replay admission do not open any referenced private payload, even after
checkpoint reclamation. Metadata existence alone neither protects state from
deletion nor proves availability. Actual execution/restoration must establish
those separate contracts at its own admission boundary.

The managed-owned projection adapter journals current-record mutations and feeds
their exact bytes to Core's generic `record` projection. Core never imports the
managed owner. A cold rebuild with managed authority requires that owner adapter
to be loaded; unsupported owners fail visibly. Full `lifecycle` relation coverage
remains explicitly unavailable pending the owning association/result integrations;
an empty relation is not advertised as supported coverage.

## Worker diagnostics

The accepted same-host Execute/Core worker diagnostic behavior is unchanged:
bounded **unredacted raw tracebacks** are the default for worker failures and can
contain exception text, source lines, local paths, private inputs or credentials.
Use `raw_traceback=False` on the selected backend for secret-bearing work. This
accepted diagnostic exception does not change other metadata/logging protection,
make dispatched code untrusted, or imply a security sandbox.
