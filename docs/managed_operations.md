# Managed Operations

`dryml.managed` provides a synchronous, checkpointed lifecycle for ordinary
methods that mutate DRYML `Object` state. It is lazy at `dryml.managed`:
importing `dryml` does not import this package. Managed does not submit work,
return a job, start a thread, import dispatch/execute/records, or import optional
frameworks.

## Public API

Declare an exact synchronous instance method with a required keyword-only
`managed` parameter. `@managed_operation(resumable=True)` permits an unfinished
compatible attempt to restore Object state and enter the current method body
again. Coroutine, generator, static/class, property, wrapped, and arbitrary
callable targets are rejected; `**kwargs` does not substitute for the slot.

```python
from dryml import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedInterrupted, managed_operation


class Counter(Pickleable):
    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def advance(self, amount, *, managed):
        self.value += amount
        managed.checkpoint()  # An author-selected safe point.
        if self.value >= 10:
            managed.interrupt()
        return self.value


store = DirStore("./counter-store")
counter = Counter(repo=Repo((store,)))
try:
    counter.advance(2, managed=ManagedConfig(state_repo=store))
except ManagedInterrupted:
    raise
```

The bound operation takes ordinary arguments plus reserved
`managed: ManagedConfig | None`; method code receives a private
`ManagedContext`, not the configuration. `ManagedConfig` accepts a borrowed
`Repo` or existing `Store` as `state_repo`, optional exact `DirStore`
`control_store`, exact-bool `rerun`, and `callbacks`
as `None` or a list of at most 64 callables. It is immutable, and invocation
snapshots its caller-owned callback list without mutating or probing it.

Bound operations expose `status(*, state_repo=None, control_store=None)` and
`request_interrupt(*, state_repo=None, control_store=None,
expected_attempt_id=None)`. Status does not run workload hooks and reports an
immutable `ManagedStatus`: `not_started`, `running`, `interrupted`, `failed`, or
`completed`. A request returns an immutable `InterruptRequestResult` with
`requested`, `already_requested`, `not_running`, or `stale_attempt`; publication
does not promise that another checkpoint will occur.

The invocation-only context exposes read-only `state_repo`, `control_store`,
`operation_id`, `attempt_id`, `is_resuming`, and `checkpoint_state_ref`.
`state_repo` is always the resolved borrowed `Repo`; a supplied `Store` is
therefore exposed through its temporary one-Store Repo wrapper, never directly.
`checkpoint()` publishes/associates a complete StateRef, while
`interrupt(cause=None)` performs that safe point then raises `ManagedInterrupted`
and chains an optional cause. Contexts cannot be caller-constructed, reused after
exit, transferred across a thread/process, or entered recursively.

The package exports `ManagedConfig`, `ManagedContext`, `ManagedStatus`,
`InterruptRequestResult`, `managed_operation`, and the `ManagedError` hierarchy.
`argument_digest` and `operation_digest` are deterministic lifecycle identity
helpers for integrations that need the same public operation addressing.
Argument identity accepts only its documented closed typed grammar and enforces
its 1 MiB complete SHA-256 preimage limit, including canonical domain and
container framing, before emitting a digest representation.

## Lifecycle And Recovery

The persistence layer borrows an explicit `Repo`, wraps an explicit existing
`Store` in a call-owned one-Store Repo, or uses the configured current Repo.
Absent current authority fails without automatic storage creation or last-state
Store selection. Omitted control authority uses the selected Repo's
`default_store`, resolved once at operation entry. Resolution validates the
physical state topology and control capability before mutation and never copies
state, migrates control, or records a control locator.

Managed control authority is a closed, bounded JSON layout rooted at
`managed/` in the selected control Store. It uses a versioned format gate,
generation-numbered current snapshots, a stable control lock, staged initial
operation directories, and a transient pending replacement intent. Inspection
does not initialize an absent namespace. A malformed gate, missing current file
inside an existing operation directory, incomplete referenced StateRef, or
pending replacement is a recovery/control error rather than a fabricated status.
Pending intents and replacement snapshots are first flushed as same-directory
temporary files, then atomically published. POSIX publication also synchronizes
the parent directory; on Windows the adapter synchronizes regular files and uses
atomic replacement, without claiming unsupported directory-descriptor fsync.

Managed lifetime ownership now composes core's process-local exact live-graph
reservation with nonblocking `dryml.locking` leases for every stateful ObjectId.
The selected state Store owns `managed/locks/v1/<hh>/<digest>.lock`; its managed
format gate is bootstrapped through the same Store-writer protocol even when
control uses another Store. Lock paths normalize a physical DirStore root, so
separate handles and control Stores cannot bypass active overlapping state
ownership. Acquisition is all-or-nothing, releases partial leases in reverse
order, and is retained through the future invocation lifetime without holding a
Store writer or control lock over workload hooks. A non-mutating complete-lock
probe is available to status/request handling: contention is inconclusive and
never itself classifies an owner as dead.

Managed methods now run synchronously in the caller's thread. Invocation requires
a materialized DRYML `Object` receiver and enters the runtime materialization
admission barrier, so strict orchestrator mode cannot execute a raw managed
method through descriptor binding. Invocation
validates the configured Stores, normalized arguments, and retained receiver graph
before ownership or control mutation.  It then retains graph and state-Store
ownership through exact resume restoration, workload execution, deep final save,
and completed-current publication.  The ordinary method result returns only after
that completion record commits. Terminal completion and failure reread the exact
active attempt and owner under the control lock, retaining a concurrent request's
generation changes instead of writing stale authority. A late external request
without another method safe point is cleared as unhonored by normal completion.
An indeterminate completion association withholds the ordinary result and remains
pending until exact reconciliation. A method error is recorded best-effort and
re-raised unchanged; an escaping `KeyboardInterrupt` becomes `ManagedInterrupted`
without an unsafe save.

Every checkpoint and final state uses the same core `Repo.save_object` engine
with the selected Repo's routing policy, deep capture, exact reservation, and
Store report enabled. Managed does not override routes with an explicit Store,
main reference, or closure mode. Before control association, it commits only
dirty buffered Stores required by all selected independent replica destinations
and the report's deterministic sufficient exact-recovery closure; unrelated
configured dirty Stores are not flushed. A required buffered source can include
reused local state or an embedded StateRef even when this save writes no new
payload there. Path-backed ZipStore authority is reopened and the exact complete
root/descendant closure and every selected replica record are verified before
association. A commit failure leaves core `StoreReport` evidence on a
`RepoSaveError` whose cause chain includes `ManagedPublicationError`; no
checkpoint/final association is published. Buffered ZipStore commits can include
earlier buffered work already present in that required Store.

`operation.status(...)` projects only the caller-selected authority, never runs
hooks or activates context. It reads immutable `ObjectRef` lock identities, so
metadata inspection remains available after a failed restore invalidates the live
target. An absent selected operation is `not_started`; a running snapshot whose
complete lock set is free across an unchanged generation is reported as a
read-only `failed`/`owner_lost` observation. Status and `request_interrupt()`
hold the short control lock across precondition checks, the nonblocking probe,
and generation reread; adapter failures remain errors rather than owner-loss
evidence. A different explicit control Store intentionally has independent,
possibly absent lifecycle metadata; managed maintains no locator.

Default invocation of compatible unfinished work resumes its retained attempt
after restoring the supplied live graph from its associated checkpoint.  Completed
or incompatible unfinished work requires `ManagedConfig(rerun=True)`.  Rerun
creates a new attempt from the current valid live Object state and does not reset
it.  Changed method code has no revision gate: restoration and workload errors
propagate normally.  A target invalidated by failed exact restoration cannot be
entered again; load fresh state from the retained checkpoint first.

Recovery, status, interruption requests, and resume resolve retained StateRef
digests across the caller-supplied Repo rather than current route selection.
They require one non-conflicting sufficient exact closure, not every historical
replica, and reject missing or conflicting retained authority.

Methods receive a private active `ManagedContext` with read-only selected Stores,
operation/attempt IDs, resume flag, and associated checkpoint reference.
`checkpoint()` deep-saves and associates the complete state, then invokes ordered
callbacks outside control locks before servicing a request that was durable at
that boundary. A callback failure retains the associated checkpoint and records
`callback_error`; a checkpoint save or association failure invokes no callbacks
and records `publication_error`. Catching either failure in method code cannot
convert the attempt into completion: managed re-raises the original safe-point
error and withholds final publication. A request arriving after the post-callback
decision waits for another checkpoint.

`interrupt(cause=...)` validates its optional exception cause before saving,
then follows the same checkpoint/callback boundary and commits `interrupted`
before raising `ManagedInterrupted` chained from that cause. A committed local
interruption remains terminal even if user code catches it: the context becomes
inactive, completion is withheld, and further context safe points are rejected.
Other public `ManagedInterrupted` exceptions, including ones from a disjoint
nested managed call, are ordinary method or callback failures for the enclosing
operation. An escaping `KeyboardInterrupt` publishes `interrupted` with the last
associated checkpoint, does not save arbitrary mutated live state, and raises
`ManagedInterrupted` chained from the original interrupt. Ordinary and callback
exceptions are re-raised unchanged after failure recording; when that recording
is indeterminate or fails, an actionable managed control error is chained from
the original exception. `SystemExit` and other non-interruption base exceptions
receive best-effort cleanup and are re-raised unchanged.

## Callback And Store Responsibilities

Callbacks receive `(live_object, context)` synchronously in supplied-list order
only after both state publication and control association succeed. A callback may
inspect `context.checkpoint_state_ref` and save a separate disjoint result through
`context.state_store`. Callbacks are trusted developer code and must not mutate
the managed object or context. Their exception stops later callbacks, retains the
checkpoint, records failure best-effort, and propagates; managed neither retries
them nor rolls back external effects. Final completion is not a callback boundary
and cannot honor a request that arrived after the last checkpoint decision.

| Concern | DRYML behavior | Caller responsibility |
| --- | --- | --- |
| State Store | Publishes complete checkpoint/final Object-state graphs. | Supply a usable `DirStore` or unambiguous current-Repo authority. |
| Control Store | Stores bounded status, attempts, requests, and StateRef associations. | Retain explicit control-store location; managed has no locator. |
| Omitted Store | Selects the sole current-Repo Store or unique matching current StateRef Store; control defaults to state. | Pass `state_store` for absent or ambiguous discovery. |
| Local ownership | Combines core reservations with `dryml.locking` advisory leases. | Use supported local filesystem semantics and handle conflict/recovery errors. |
| Platform matrix | Exercises the current in-house environment and portable adapter seams. | Do not infer Windows, network, distributed, or multi-host certification. |

Control data does not change CDef, ObjectRef, or StateRef identity. Separate
control storage does not copy checkpoint payloads. A different explicit empty
control Store may start fresh work after ownership ends because managed has no
global operation locator, journal, or duplicate-work prevention.

Managed control is closed canonical JSON format v1: gate `dryml-managed`, current
snapshot `dryml-managed-current`, and replacement intent `dryml-managed-pending`.
Malformed, pending, incomplete, unreadable, or unsupported control authority,
including a non-directory operation-path ancestor, is a recovery or control
error, never absent or successful work. Object state remains the closed
DirStore v2/StateRef authority documented in [Formats](formats.md); there is no
migration, compatibility reader, or separate continuation payload.

Resume restores Object state, not a Python stack or unsaved external resources.
Exact receiver identity, normalized arguments, and the current resumable
declaration must match. There is deliberately no authored version, capability
digest, source fingerprint, or semantic code-compatibility guarantee. Changed
code may run or raise from restore/method code; silent semantic mismatches are
not detected.

Restore preflight happens before hooks. A hook failure after restore begins
invalidates the supplied live target graph for state IO, resume, and rerun. Its
immutable checkpoint remains authoritative: inspect explicit Store status, load a
fresh exact graph with `Repo.load_state_ref(checkpoint, reuse_live="never")`, and
resume that object. Default `Pickleable` restoration replaces ordinary payload
fields, removing stale fields absent from the checkpoint while preserving graph
bindings and framework runtime metadata.
