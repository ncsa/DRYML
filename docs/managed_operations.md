# Managed Operations

`dryml.managed` begins with checked declaration and input-matching support for
synchronous instance methods. Use `@managed_operation(resumable=...)` with a
required keyword-only `managed` parameter, and pass caller policy through
`ManagedConfig`.

The internal U4 persistence layer resolves explicit `DirStore` bindings, or an
omitted state Store from `dryml.core.session.current_repo`: one physical Store is
selected directly, while multiple physical Stores require exactly one full match
for the live object's exact current `StateRef`. Omitted control authority uses
the selected state Store. Resolution validates publication capabilities before
mutation and never copies state, migrates control, or records a control locator.

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
format gate is bootstrapped through the same U4 Store-writer protocol even when
control uses another Store. Lock paths normalize a physical DirStore root, so
separate handles and control Stores cannot bypass active overlapping state
ownership. Acquisition is all-or-nothing, releases partial leases in reverse
order, and is retained through the future invocation lifetime without holding a
Store writer or control lock over workload hooks. A non-mutating complete-lock
probe is available to U6 status/request handling: contention is inconclusive and
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

Methods receive a private active `ManagedContext` with read-only selected Stores,
operation/attempt IDs, resume flag, and associated checkpoint reference.  U7 owns
checkpoint callbacks and interruption safe points, so `checkpoint()` and
`interrupt()` explicitly reject use until that protocol is implemented rather
than publishing partial lifecycle state.
