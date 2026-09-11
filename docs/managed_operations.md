# Managed Operations

`dryml.managed` provides a synchronous, checkpointed lifecycle for methods that
mutate DRYML Object state. It is lazy at `dryml.managed`: importing `dryml` or
`dryml.core` does not import managed, execute, or dispatch. Managed does not
submit work, start threads, provision environments, or sandbox code. It operates
on trusted same-host Objects and Stores using local filesystem locks.

## Declaring And Calling

Decorate an exact synchronous instance method with `@managed_operation`. The
method must declare a required keyword-only `managed` parameter. Coroutines,
generators, static/class methods, properties, wrapped targets, and arbitrary
callables are rejected. `@managed_operation(resumable=True)` permits a compatible
unfinished attempt to restore its retained Object state before re-entering the
method body.

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
        managed.checkpoint()
        if self.value >= 10:
            managed.interrupt()
        return self.value


state = DirStore("./counter-state")
control = DirStore("./counter-control")
repo = Repo([state])
counter = Counter(repo=repo)
try:
    counter.advance(2, managed=ManagedConfig(state_repo=repo, control_store=control))
except ManagedInterrupted:
    # Inspect or resume using the same state Repo and selected control Store.
    raise
```

The bound method accepts ordinary arguments plus `managed: ManagedConfig | None`.
Method code receives a private `ManagedContext`, never the configuration itself.
`ManagedConfig.state_repo` accepts a borrowed `Repo`, an existing borrowed
`Store`, or `None`. A Store is wrapped in a private one-Store Repo with ordinary
unconfigured closure behavior; paths and Store lists are not managed inputs.
`None` uses the configured current Repo and fails when none is configured. The
selected Repo is retained intact, including its routes and replicas, and is never
reduced to the Store containing a prior StateRef.

`control_store` is an optional exact `DirStore`. When omitted, it is resolved
once from `state_repo.default_store`, not from a matching route or root save
destination. An absent, unwritable, ZipStore, or otherwise unsupported default
fails before workload execution; choose an independent supported control Store
explicitly. `rerun` is an exact bool. `callbacks` is `None` or a caller-owned list
of at most 64 callables; invocation snapshots that list without mutating it.

Bound operations also provide
`status(*, state_repo=None, control_store=None)` and
`request_interrupt(*, state_repo=None, control_store=None, expected_attempt_id=None)`.
They read only caller-selected authority and do not run workload hooks. Status
returns immutable `ManagedStatus` values such as `not_started`, `running`,
`interrupted`, `failed`, or `completed`. An interruption request returns immutable
`InterruptRequestResult` status; it requests a later cooperative safe point and
does not guarantee another checkpoint.

## State, Control, And Ownership

The invocation context exposes read-only `state_repo`, `control_store`,
`operation_id`, `attempt_id`, `is_resuming`, and `checkpoint_state_ref`.
`state_repo` is the resolved borrowed Repo; a Store input is therefore visible
through its temporary wrapper. Contexts are runtime-created, cannot be
constructed by callers, reused after exit, transferred to another thread/process,
or entered recursively.

Before user code, managed validates selected state and control authority and
acquires the Cartesian set of every distinct physical state Store and every
stateful ObjectId in the exact receiver graph. The lock order is stable. Any
shared Store/ObjectId pair conflicts even when callers use different control
Stores, route orders, or Store handles; disjoint physical Store sets are
independent. Direct DirStores use an in-root managed lock namespace; a
path-backed ZipStore uses a sibling namespace derived from its archive path, not
its extraction directory. Unsupported Store identity/topology fails before user
code. Incomplete owner-probe evidence is a recovery error, never proof that an
owner is gone.

The resolved state Repo, control Store, and physical state Store set remain fixed
for an invocation. Adding, removing, replacing, or closing connected resources
while that topology lease is active fails. Later routing/default-order changes can
affect a later checkpoint but not the in-flight save or selected control Store.
Managed borrows caller Repos and Stores: it releases only temporary wrappers and
leases it created, and never closes borrowed handles.

## Checkpoints And Recovery

`ManagedContext.checkpoint()` deep-captures through the selected Repo's normal
routing policy with the invocation's exact graph reservation. It does not pass
an explicit Store or force placement/replication. `interrupt(cause=None)` follows
the same checkpoint boundary, then records interruption and raises
`ManagedInterrupted`, optionally chained from `cause`.

Before a checkpoint or final state is associated with control authority, managed
requires every routed publication and required exact-recovery dependency to be
complete. It commits every dirty buffered Store required by the report, including
reused state or embedded-reference sources, but does not blanket-flush unrelated
connected Stores. A ZipStore commit can publish earlier buffered work in that
required archive. Managed then validates each selected replica and exact recovery,
including fresh path-backed ZipStore views, before associating the StateRef in the
control Store. A publication, commit, validation, or association failure leaves
completed immutable state intact, preserves the prior associated checkpoint, and
does not report a new checkpoint or final success. Core partial `StoreReport`

Callbacks receive `(live_object, context)` in caller order only after both state
publication and control association succeed. They may inspect
`context.checkpoint_state_ref` or save an independent result through
`context.state_repo`. They are trusted developer code and must not mutate the
managed Object or context. A callback error stops later callbacks, retains the
associated checkpoint, records `callback_error` best-effort, and is re-raised;
there is no retry or rollback of external effects. Final completion is not a
callback boundary.

Status, interruption, resume, and recovery locate retained StateRefs across the
caller-supplied state Repo rather than re-evaluating current routing. One complete,
non-conflicting exact closure is sufficient; identical replicas are accepted, but
missing, conflicting, or incomplete authority raises `ManagedRecoveryError`.
Control records contain no Repo locator, Store handle, portable definition, or
state payload. After restart, callers reconnect the required state Stores and
select the intended control Store themselves.

Compatible unfinished work resumes its retained attempt by restoring the supplied
live graph. Completed or incompatible unfinished work requires `rerun=True`,
which starts a new attempt from current valid live state without resetting it.
Resume restores Object state, not a Python stack, unsaved external resources, or
semantic compatibility with changed method code. A restore-hook failure invalidates
the supplied graph for later framework state IO; load a fresh exact graph from the
retained checkpoint before retrying.

## Control Format

Managed control is separate bounded canonical JSON authority rooted at `managed/`
in the selected control DirStore. It uses the `dryml-managed` v1 format gate,
generation-numbered current snapshots, a stable control lock, staged initial
operation directories, and `dryml-managed-pending` v1 replacement intents.
Records retain lifecycle identity, state, request, ownership evidence, and
associated StateRef digests, not checkpoint payloads or a continuation. Reads do
not initialize an absent namespace. Malformed, pending, incomplete, unreadable,
or unsupported control authority is a control/recovery error, not absent or
successful work. See [Formats](formats.md) for the durable grammar.
