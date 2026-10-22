# Managed Operations

`dryml.managed` owns the lifecycle of state-mutating Object methods. Its **U14
Node identity/control foundation** supports detached metadata admission,
operation-scoped ownership, fenced control CAS, metadata history, and independent
status/interruption control. Importing Core does not import managed, Execute,
Dispatch, or optional frameworks.

## Integration status

The previous StateRef/current-v2 execution path is retired. Calling a decorated
method currently raises `ManagedControlError("lifecycle_not_integrated", ...)`
before body execution, state capture, restore, or control publication. Declarations
remain available with `@managed_operation`, including the checked synchronous
instance-method/signature composition rules.

This foundation is not the completed managed runtime. U15 owns selected starts,
two-phase rerun and resume admission policy; U16 owns checkpoint/final policy and
retention handshakes; U17 owns durable ordinary results and completed replay; U19
integrates the detached target into execution transport. In particular, acquiring
an admission or constructing control metadata does not claim that a result can
already be replayed or that its private state is retained/available.

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
receiver reconstruction, so a future completed replay need not open a reclaimed
receiver checkpoint. Recovery admissions report `reconciled=True` and refuse
realization: reconciliation terminates separately from execution.

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
- Version-1 named managed Node, history and root-intent metadata records.
- Nested `dryml-node-graph/v1` / identity-3 graph, reference and StateKey codecs.

Area format 2 does **not** admit old current-record version 2. Every owning reader
rejects incompatible nested versions as well as old area/current/pending records,
preserving source bytes. No old authority is automatically converted, deleted or
overwritten. A pending replacement is reconciled only against its exact
previous/proposed generation and byte fingerprints under operation ownership.

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
