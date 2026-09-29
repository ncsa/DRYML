# Generic Execute

`dryml.execute` runs one trusted Python callable through an explicitly
selected backend. It is the generic byte-oriented execution layer: it is not
`Repo`/`Store` transport, managed-operation execution, Dispatch selection, a
runtime/session API, or a safe-deserialization boundary. Callable graphs and
serialized payloads are trusted inputs. `dryml.core.execute` is the separate
opt-in core adapter which owns core reference transport, worker setup, result
publication, and caller recovery over this generic layer.

There is no persistent resource ledger: each executor reports only
coordinator-and-backend scoped observations.

## Callable Transport

Execute supports function roots: importable functions, lambdas, nested
functions, closures with serializable captures, and importable unbound builtins.
It does not promise that every value satisfying `Callable` is transportable.
Stateful bound methods, bound builtin methods, and callable instances are
rejected synchronously before backend submission. Known DRYML core references
are also rejected by a fixed generic transport error; use the explicit
`dryml.core.execute` adapter when a call needs core authority. The generic
rejection does not expose type-specific core phrases or the submitted value's
private details.

Use `Executor` for a reusable backend lifetime, or the explicit one-off `run`
and `submit` helpers. The common facade does not eagerly import subprocess or
Ray modules. There is no default backend, string backend selector, implicit
Ray initialization, environment creation, package installation, cluster
provisioning, fallback backend, persistent resource ledger, or cross-coordinator
accounting.

```python
from dryml.execute import Executor
from dryml.execute.subprocess import SubProcessConfig

with Executor(SubProcessConfig()) as executor:
    assert executor.run(sum, [1, 2, 3]) == 6
```

## Public API

`Executor(config)` is inert: construction captures its working directory and
interpreter but does not create a spool, inspect files, start a backend, or run
work. `start()` initializes the selected backend once; it is also called lazily
by submission, `discover()`, and `resources()`.

```python
from dryml.execute import ExecutionOutput, Executor
from dryml.execute.subprocess import SubProcessConfig

offset = 5
output = ExecutionOutput()
with Executor(SubProcessConfig()) as executor:
    future = executor.submit(lambda value: value + offset, 7, output=output)
    assert future.result() == 12
    future.cleanup()
    assert output.snapshot().complete
```

`Executor.submit(fn, /, *args, kwargs=None, environment=None,
environment_spec=None, world=None,
execution_timeout="inherit", stream_output=None, done_callbacks=(), output=None,
worker_setup=None)`
copies controls, serializes exactly one callable/argument graph to an
execution-owned spool child, and returns the backend's concrete
`ExecutionFuture`. `kwargs` is the workload keyword mapping; it is deliberately
separate from Execute controls. Invalid controls, requirements, preflight
serialization, spool capacity, or an executor closing fail synchronously.
Accepted asynchronous failures are published by the future.

`environment_spec` is an independent exact existing-environment pin, not an
`EnvironmentRequirement` and not a preference. `CurrentEnvironmentSpec` freezes
the submitting coordinator's interpreter, prefix, base prefix, and stable
Python/DRYML/distribution evidence at call entry. `PythonExecutableSpec` preserves
its supplied executable spelling, including a venv symlink. A named
`CondaEnvironmentSpec` resolves once to exactly one existing prefix; zero or
ambiguous matches fail. A supplied selector overrides `SubProcessConfig`'s
`python_executable` and candidate search, including when `environment` is `None`.
Unsupported container or backend launcher forms fail before payload transfer;
Execute never provisions software or falls back to another environment.

`Executor.run` has the same keyword-only controls and returns `future.result()`.
It does not close the reusable executor. `Executor.with_options(...)` returns
an `ExecutorView` retaining the parent executor. A view has the same `run(fn,
/, *args, **kwargs)` and `submit(fn, /, *args, **kwargs)` workload boundary, so
control-named workload keywords remain ordinary workload data. The view never
owns a second backend, quota, future set, or close operation.

`with_options(environment_spec=...)` retains the typed selector without I/O. Each
later `run` or `submit` resolves it once at that operation's entry, so a
`CurrentEnvironmentSpec` observes the then-current coordinator environment rather
than the environment at view construction. `Executor.discover(environment_spec=...)`
uses one independent resolution, reports only that frozen target, and remains
non-reserving. Generic and core module-level
`submit`/`run`, reusable `submit`/`run`, views, and `discover` accept the same
control. Core Execute forwards it to generic Execute and has no separate selector
implementation.

Selector environment variables and `pythonpath_policy` are frozen with the
operation and applied to subprocess launches. Ray passes an exact selected Python
path through its supported `py_executable` runtime control and verifies fresh
worker identity before payload transfer. Ray rejects unsupported launcher forms,
including `conda-run`, during discovery and submission rather than reporting them
as viable or dropping their launch controls.

`WorkerSetup(factory="module:qualname", data={...})` is an optional immutable
generic control for a trusted worker-local context manager. The factory identifier
is bounded and importable-shaped; `data` is detached JSON only. After verified
admission and `GO`, Execute sends the bounded setup/evidence envelope, captures
setup output, enters the factory, waits for `SETUP_READY`, and only then sends the
callable payload. Calls without `worker_setup` retain the direct `GO`-to-payload
sequence. Setup entry failure withholds the payload. Setup exit covers result
encoding; an exit failure preserves an already encoded result or workload error,
but leaves the Future cleanup state incomplete and `cleanup()` raises rather than
claiming that unobserved worker teardown was reconciled.

The generic private worker protocol is version 4. After admission and `GO`,
setup-bearing calls send `SETUP` and wait for `SETUP_READY` before `PAYLOAD`.
Bounded `OUTPUT` may arrive during setup, invocation, and teardown; terminal
outcomes and final output fences complete the exchange. An incompatible worker fails before `GO`;
`RESULT` before `PAYLOAD`, or payload after setup failure, is invalid. The
post-`GO` `execution_timeout` covers setup, payload transfer, invocation, outcome
encoding, and normal setup exit. It is independent of `admission_timeout` and
`output_final_timeout`; the configuration matrix below gives their defaults and
other limits.

If the worker observes that deadline at either explicit check immediately before
payload deserialization or callable invocation, its private bounded `ERROR`
terminal carries a closed pre-invocation deadline marker. The marker is distinct
from the remote exception type field and is preserved inside setup-bearing
terminals together with setup cleanup evidence. A callable-raised `TimeoutError`
therefore remains an ordinary `RemoteExecutionError`, even if the wall clock has
expired by the time the coordinator receives it. The first validated ordinary
result/error or deadline/cancellation claim wins. A deadline marker does not by
itself prove completion: subprocess reports `ExecutionDeadlineExceeded` only
after owned-group termination is confirmed, Ray only after its native task or
worker lifetime is qualified, and either backend reports uncertainty when that
lifecycle evidence cannot be obtained. Output final fences remain independent
and are drained under their existing bound.

The core adapter's `dryml.core.execute:core_worker_setup` is a worker setup
factory. It publishes runtime controls, activates the public Session resource
cache, then reopens a detached `RepoDefinition` and optional control Store through
that cache before temporarily installing `core.session.config` and
`current_context()`. The context exposes only the worker's borrowed Repo and
optional explicitly selected control Store. It is task/thread owned, is
unavailable outside that scope, and does not modify caller session state.
When setup has an explicit control Store, core also installs a private
task/thread-owned managed control default before invocation decoding. Managed
methods still resolve an omitted `state_repo` from invocation-time `current_repo`,
so a supported nested session override is honored; an explicit config state or
control field wins. Without a setup control role, omitted control falls back to
the effective state Repo's default Store. This private default expires with setup
and cannot be retained by copied task or thread contexts.
Equivalent reconstruction during the workload reuses setup handles while the
setup scope is active. On exit, worker/session contexts restore before the cache
closes cache-owned handles with `flush=False`; runtime activation remains in
effect through cleanup and one resource-close retry. If closure still fails, the
worker reports `worker_setup_exit_failed` without replacing a workload failure.
It retains the failed cleanup owner until process exit and rejects further core
setup in that process with exit/restart guidance. Current subprocess and Ray
workers are one-shot; no workload is retried. A control Store shared with the Repo uses
its existing table handle, while a separate direct directory Store is cache-owned
by the same worker scope. This does not change generic Execute RPC/setup fields
or make generic Execute responsible for managed-operation policy. The shared
worker strategy still rejects ZipStore transport even though standalone Session
caching can locally retain a ZipStore.

`dryml.core.execute.CoreOptions` is an inert reusable core-adapter override.
At submission preparation, `resolve_core_options()` resolves per-call settings
over reusable executor settings over the current core session's Repo/cache
defaults. `"inherit"` falls through one layer, while `None` explicitly clears a
Repo, control Store, or runtime selection. Runtime inheritance ends at `None`:
workers use their established INLINE baseline rather than a clone of caller
runtime state. In orchestration mode, requests for live returned Objects or
argument updates fail before any Store export, opening, mutation, or backend
submission.

The core adapter supports ordinary functions, lambdas, nested functions and
closures, bound methods, callable instances, `@function` callables, `Method`, and
managed-operation callables. It sends one whole callable/argument/capture graph,
including globals, defaults, annotations, instance fields, and `__slots__`.
Stable captured API function dependencies owned by `dryml.*`, such as process-local
context getters, use their import reference. Caller-owned helper functions,
including importable module-level helpers, and closure values remain structurally
captured with their coordinator globals and defaults. Standard `pathlib` values
use their public kind and text rather than version-specific private slots.
Stable DRYML-owned managed declarations likewise import the authored and
executable functions through their exact owner member; caller-owned declarations
remain captured by value.
Live `Repo` and `Store` captures are rejected except in an exact
`ManagedConfig` value. Core invocation graph v2 captures those configs wherever
they occur in supported arguments, containers, defaults, captures, or instance
fields, snapshots the exact `rerun` bool and callback-list membership, and
preserves repeated config/callback aliases. It transports only detached direct
`DirStore` descriptors and detached Repo definitions containing direct
`DirStore`s; ZipStore and arbitrary live resource captures remain rejected.
Callback policy preserves `None` separately from an explicitly empty callback
list. Managed configuration is invocation-only: it cannot appear in a result
graph, so a completed result never returns a live reconstructed resource.
The same closed v2 graph represents managed declarations, composites, and bound
targets with authored function, executable wrapper, member, and recognized owner
evidence. Nested callbacks may capture a managed declaration or its
default/closure values without relying on member lookup as graph evidence.
Explicit config nodes reopen selected existing authority while decoding through
the active worker Session resource cache. Thus matching config resources reuse
the worker's handles, while different supported settings remain distinct. A
malformed or unavailable selected resource fails at decode before workload
invocation; an ordinary descriptor captured only by an untaken code branch is
not opened until that code requests it. Core values otherwise lower to exact CDef,
`ObjectRef`, or `StateRef` authority; a selected declaration Store is a pinned
index in the frozen Store table, never a path. The worker reconstructs one local
owner boundary: function, Method, and managed owners each deliver/invoke/normalize
once rather than gaining a second generic signature boundary.

Core Execute does not inject a root `ManagedConfig` into a managed target. Direct
managed targets and managed calls nested inside ordinary transported callables use
the same managed resolver. Worker failure, cancellation, and lost delivery remain
ordinary Execute/core errors; the coordinator does not inspect managed status or
automatically reconcile, retry, resume, or rerun mutation.

A same-host worker may compute or republish a CachedDataset when its exact source
and state authority are available through the supported shared direct-DirStore
strategy. The managed call returns its final StateRef as reference data; payload
chunks and live Dataset/Repo/Store objects are not result transport. Same-codec
completed content can republish without opening the source. A changed codec or
unfinished resume still needs the retained source/work authority. Missing or
malformed descriptors, worker interruption, publication failure, result delivery
failure, or incomplete cleanup retain their ordinary Execute/managed failure
classification and never become a successful cache result. Cross-host Stores,
automatic remote refresh, and runtime provisioning remain unsupported.
See [Managed Operations](managed_operations.md) for lifecycle recovery,
[Session](session.md) for resource-cache lifetime, and [Repos and Stores](repos.md)
for the authoritative state and detached-definition boundaries.

`prepare_shared_storage()` is the core-call storage seam. It exports a live Repo
exactly once, derives the full worker Store table
and control role from that one detached `RepoDefinition`, and retains a fresh,
submission-owned recovery Repo. The initial `SharedDirStoreStrategy` accepts
only existing direct `DirStore` authority; it rejects absent storage, ZipStore
descriptors (including extracted archive handles), inaccessible directories,
ambiguous physical destinations, and unsupported strategy identities. A control
Store shared with the Repo is encoded as its table index; a separate Store uses
an explicit direct-directory descriptor. Snapshot cleanup uses `flush=False` and
never closes caller-borrowed handles. Callable payload preparation and generic
submission proceed through the prepared call codec.

### Core Executor Facade

`dryml.core.execute.Executor(config, core=None)` owns one generic `Executor`
and exposes core-aware `submit`, `run`, `with_options`, `discover`, `resources`,
`start`, and `close` methods. Its configuration is the same explicit public
`BackendConfig` used by generic Execute, so configured deadlines, framing limits,
output limits, polling, and cleanup budgets are forwarded to the worker setup,
invocation, outcome, and recovery path without a core-local override.

`Executor.submit(fn, /, *args, kwargs=None, core=None, environment=None,
environment_spec=None, world=None, execution_timeout="inherit", stream_output=None,
done_callbacks=(), output=None)` returns `CoreExecutionFuture`, not the generic
byte Future. Per-call core options are resolved over executor options and the
submission caller's session once before callable preparation; the Store table,
runtime, cache policy, result materialization decision, and caller refresh
targets cannot be changed by later caller/session mutations. `"auto"` result
materialization is decided in that submission caller, not the adaptation thread.

`CoreExecutionFuture.backend_future` is borrowed advanced access to the public
generic `ExecutionFuture[bytes]`; its result is internal outcome bytes. Use the
core facade's `result`, `exception`, awaiting, callbacks, `snapshot`, and
`cleanup` methods for the recovered result. One public generic completion callback
per submission performs recovery and requested argument refresh exactly once.
Concurrent waiters, callbacks, and awaiters join that adaptation. Caller wait
timeouts do not cancel backend work or recovery. Core callbacks receive the core
facade after adaptation and their exceptions cannot change its terminal result.

`CoreExecutionSnapshot` contains the nested generic snapshot, core adaptation
state/phase, detached `CoreOutcomeEvidence` and refresh ledger, plus independent
cleanup state/issues. `cleanup()` rejects before core terminality, then joins
generic cleanup and closes only the facade's reconstructed recovery Repo with
`flush=False`. A later cleanup failure remains observable and raises
`CleanupError`, but does not rewrite a successful recovered result or close a
caller-borrowed Repo/Store.

`Executor.with_options(...)` returns `ExecutorView`, which binds core and generic
controls without a second backend. Every keyword passed to a view's `run` or
`submit` is workload data, including names that collide with executor controls.
Module-level `dryml.core.execute.submit(..., backend=...)` and `run(...,
backend=...)` require an explicit backend and use the generic bounded one-off
owner; their core recovery resources remain independently owned by the facade.

### Core Results

`SharedDirStoreStrategy` transports one bounded tagged outcome rather than a
pickled live Object or `StoreReport`. A live returned Object is saved through the
worker Repo's normal flushing `save(..., deep_capture=True, report_stores=True)`
path before shared automatic reference selection. Stateful graphs therefore
return a `StateRef`; stateless graphs return their CDef; incoming CDef,
`ObjectRef`, and `StateRef` result values remain reference data.

Execute owns result publication before requesting shared-signature `AutoRef`
selection; signature selection itself never saves an Object. The core codec
is a bounded, version-local worker/coordinator implementation detail. It is not a
Store format, general pickle format, or a cross-version RPC compatibility promise.

With `update_args=False` (the default), execution never saves merely-mutated
arguments and result recovery never restores into an original caller Object.
With `update_args=True`, explicitly materialized live argument graphs are
coalesced to maximal roots, saved once, and restored into their original caller
instances after all delivered update references preflight successfully. A failed
restore leaves previously applied targets intact and does not replay execution.
Overlapping returned roots and descendants reuse the update StateRef, with a
descendant represented by `root_state_ref.at(path)`.

The worker validates the complete result and selected update graph before any
publication. It coalesces each graph to maximal roots and reuses the one saved
snapshot for a returned root or descendant and its matching update. It does not
promise an atomic result/update transaction: completed authoritative publications
remain visible after a later failure, and refresh failure neither rolls back
earlier restores nor replays the workload.

`decode_core_outcome()` exposes `CoreAdaptationOutcome` and
`CoreOutcomeEvidence` value types. Evidence has exact StateRefs and
Store-table-relative publication status only; it never carries a live Repo,
Store, `StoreReport`, argument, or result. A delivered worker failure preserves
known publication evidence but does not imply rollback, retry, or successful
caller refresh.

`PreparedCoreCall` contains only invocation bytes, frozen storage setup, and
opaque update descriptors. It never retains caller Objects or refresh progress.
The future-facing coordinator binds `SharedDirStoreStrategy.bind_recovery()`
immediately after preparation, before caller mutation, and retains that private
state for recovery. Direct `recover()` remains supported by binding its supplied
arguments for that one call.

Before a worker saves an update or result, it validates the complete result and
selected update graphs and reserves the configured result bound for every
possible StateRef and Store-table publication fact. If a result cannot fit after
publication, the failed outcome drops only result bytes and retains exact
publication/update evidence; it never replaces that evidence with a generic
failure.

Its `WorkerSetup.data` is exactly one self-validating envelope:

```json
{
  "contract_version": "1.1",
  "schema": "dryml.core.execute.v1.1",
  "kind": "worker_setup",
  "payload": {
    "runtime": "dryml.runtime.v1.1 runtime_context envelope",
    "repo": "dryml-repo-definition envelope",
    "role": "main",
    "replica": 0,
    "control_store": null,
    "cache": "weak"
  },
  "id": "core_setup-v1.1-<sha256>"
}
```

`payload` has no aliases or extra fields. Core decoding remains inert until both
nested owner envelopes validate; only then can runtime activation precede Store
reconstruction. A subprocess setup without a world allocation receives a
baseline grant with no invented CPU IDs, allocation identity, affinity, or
memory control. Ray receives only its verified logical CPU/memory quantities and
accelerator IDs. Framework thread requests belong to a registered framework's
runtime plan and must not exceed reported logical capacity; generic thread-limit
environment controls are rejected.

Module-level `submit(fn, /, *args, backend=..., ...)` and `run(...,
backend=..., ...)` require a `BackendConfig`; they retain a hidden owner until
the future's cleanup completes. One-off automatic cleanup uses the configured
bounded attempts and retry interval. A cleanup failure preserves the future and
its result/error for explicit recovery rather than reporting cleanup as complete.

`Executor.discover(environment=None, environment_spec=None, world=None, timeout=None)` returns a
non-reserving `DiscoverySnapshot`; `resources(timeout=None)` returns a bounded
backend-scoped `ResourceSnapshot`. `close(cancel=False, timeout=None)` first
stops acceptance, then reconciles accepted work, spools, backend handles, and
the executor lease. `cancel=True` requests cancellation for outstanding work.
`CleanupError` means the executor or future remains inspectable and retryable.

`ExecutionFuture.result(timeout=None)`, `exception(timeout=None)`, `done()`,
`running()`, `cancelled()`, and `snapshot()` provide the common result and state
interface. A wait timeout does not cancel work. `cancel()` only confirms a
pre-GO cancellation; `request_cancel()` asks the backend to stop running work
but does not claim it succeeded. `cleanup(timeout=None)` is allowed only after a
terminal outcome and frees only that submission's resources. Completion
callbacks receive the exact concrete future, are registered before dispatch for
`done_callbacks`, execute outside locks, and cannot alter the outcome. Callbacks
added after terminality are also scheduled asynchronously. Futures are awaitable.

`ExecutionOutput` is an optional single-use caller-owned output holder. Its
snapshot retains bounded stdout/stderr prefixes, truncation flags, final-fence
completion, and a best-effort live-delivery issue. Output remains available after
future cleanup. A missing final fence after `output_final_timeout` seconds makes
capture incomplete without rewriting a validated workload result.

`ExecutionError` is the bounded base failure and may retain immutable admission
evidence. `AdmissionError` reports rejected environment, world, or backend
admission; `BackendUnavailableError` reports an unavailable selected backend;
and `RemoteExecutionError` reports a bounded worker type without reconstructing
its exception class. This includes a user callable's own `TimeoutError`.
`ExecutionDeadlineExceeded` means the worker or coordinator observed the
execution deadline and owned termination was then confirmed; a worker marker
alone is insufficient. `ExecutionUncertainError` means available
evidence cannot safely classify success. `CleanupError` retains the recoverable
future when backend/spool cleanup is incomplete.

The immutable observation values are `ExecutionIssue`, `AdmissionReport`,
`ExecutionSnapshot`, `OutputSnapshot`, `EnvironmentCandidate`, `FeasiblePlan`,
`DiscoverySnapshot`, `ResourceAmounts`, `ActiveAllocation`, and
`ResourceSnapshot`. Resource amounts and snapshots describe coordinator-known,
backend-scoped logical capacity and charges, not physical isolation, a global
machine inventory, or another coordinator's ledger.

## Configuration

`BackendConfig` is an abstract frozen keyword-only dataclass. Its constructor
performs no I/O. Durations are finite positive seconds; `execution_timeout` may
also be `None`. Count and byte limits are positive non-boolean integers unless
noted otherwise. Invalid types, non-finite durations, contradictory limits, and
unsupported values raise `TypeError` or `ValueError` before backend work.
Relative `working_directory`, `spool_directory`, and direct interpreter paths
are resolved against the coordinator's construction CWD by `Executor`; this
capture does not probe or create them.

| Setting | Default and validation | Effect and override |
| --- | --- | --- |
| `admission_timeout` | `30.0` seconds, positive finite float/int | Bounds preflight/backend admission for each call. It ends at `GO`; it never times post-GO setup. |
| `discovery_timeout` | `30.0` seconds, positive finite | Default for `discover()` and `resources()`; each accepts a positive `timeout=` override. |
| `termination_timeout` | `5.0` seconds, positive finite | Default future cleanup/cancellation bound; `close(timeout=)` and `cleanup(timeout=)` can supply a positive override. |
| `one_off_cleanup_attempts` | `2`, positive integer | Bounded automatic retries for hidden one-off owners. |
| `one_off_cleanup_retry_interval` | `0.1` seconds, positive finite | Cadence for retained one-off cleanup retries. |
| `execution_timeout` | `None`, or positive finite seconds | One deadline for post-GO setup, payload transfer, invocation, result encoding, and normal setup exit. Per call uses `"inherit"`, `None`, or a positive override; `None` disables that deadline. |
| `output_final_timeout` | `5.0` seconds, positive finite | Seconds to wait after terminal outcome for output final fences; it changes output completeness only. |
| `spool_directory` | `None`, or `pathlib.Path` | Parent for coordinator-owned spool children. `None` chooses the platform temp parent. The executor creates/removes only its child data and never deletes a caller parent; Windows children and files receive a verified private ACL before payload bytes are written. |
| `spool_limit_bytes` | `4,294,967,296`, positive integer bytes | Process-global aggregate spool quota across all active executor leases, backend types, and spool parents; must cover invocation plus result limits. |
| `spool_file_limit` | `128`, integer at least `2` | Process-global file reservation limit across all active executor leases, backend types, and spool parents; reserves invocation and result files. |
| `preflight_limit` | `8`, positive integer | Process-global maximum concurrent payload preflights across all active executor leases, backend types, and spool parents. |
| `invocation_limit_bytes` | `67,108,864`, positive integer bytes | Maximum serialized callable/argument payload and wire frame. |
| `result_limit_bytes` | `67,108,864`, positive integer bytes | Maximum serialized result and result wire frame. |
| `control_header_limit_bytes` | `1,048,576`, positive integer bytes, at most 4-byte frame range | Maximum protocol control header; cannot exceed `admission_message_limit_bytes`. |
| `owner_envelope_limit_bytes` | `16,777,216`, positive integer bytes | Maximum encoded environment/world/SETUP owner envelope. |
| `admission_message_limit_bytes` | `83,886,080`, positive integer bytes | Maximum bounded admission message, including one SETUP envelope and its backend evidence. |
| `output_frame_limit_bytes` | `65,536`, positive integer bytes | Maximum worker output frame; cannot exceed live queue capacity. |
| `output_limit_bytes` | `1,048,576`, positive integer bytes | Retained byte prefix limit for each output stream. |
| `live_output_queue_limit_bytes` | `262,144`, positive integer bytes | Bounded coordinator live-delivery queue; overflow disables mirroring but preserves retained capture. |
| `diagnostic_text_limit_bytes` | `65,536`, positive integer bytes | Bounded retained framework diagnostic text. |
| `diagnostic_issue_limit` | `64`, positive integer | Maximum retained cleanup diagnostics. |
| `discovery_candidate_limit` | `128`, positive integer | Bound on discovered environment candidates. |
| `discovery_directory_entry_limit` | `1,024`, positive integer | Bound on entries examined in each discovery directory. |
| `process_read_chunk_bytes` | `8,192`, positive integer bytes | Bounded chunk size while reading owned probes/process output. |
| `process_poll_interval` | `0.005` seconds, positive finite | Poll cadence for owned process and bounded observation work. |
| `environment_search_depth` | `2`, nonnegative integer | Maximum discovery traversal depth; `0` disables descent below roots. |
| `stream_output` | `False`, bool | Default per-call live output choice; `submit(..., stream_output=)` overrides it. |
| `automatic_environment_discovery` | `True`, bool | Enables bounded candidate discovery when discovery or an environment requirement needs it. |
| `conda_executable` | `"conda"`, nonempty string | Default existing Conda launcher selector; construction does not resolve it. |
| `conda_launch_mode` | `"direct"`, `"direct"` or `"conda-run"` | Default existing Conda launch form. A supplied `CondaEnvironmentSpec` retains its own form. |
| `environment_candidates` | `()`, tuple of `EnvironmentSpec` | Explicit existing candidates, copied/frozen at construction; no candidate is provisioned. |
| `environment_search_roots` | `()`, tuple of `Path` | Additional bounded filesystem discovery roots. Reading is deferred until discovery/submission needs it. |
| `working_directory` | `None`, or `Path` | Captured CWD when absent; selected worker receives it as its working directory. It is not created by configuration. |
| `env_vars` | `{}`, string-to-string mapping | Frozen explicit worker environment overrides. They are passed to an owned subprocess or selected existing runtime; do not put credentials in diagnostics. |

| Specialized setting | Default and validation | Effect and override |
| --- | --- | --- |
| `python_executable` | `None`, or `pathlib.Path` | `SubProcessConfig` captures the coordinator interpreter when absent; a supplied path is checked at launch, not config construction. |
| `address` | `"auto"`, or existing `host:port` text | `RayBackendConfig` never accepts a URI or provisioning form; explicit endpoints never fall back to discovery. |
| `namespace` | `None`, or nonempty string | Optional existing Ray namespace that must match a borrowed caller connection. |
| `connect_timeout` | `30.0` seconds, positive finite | Bounds a caller waiting for Ray initialization, not an uninterruptible SDK initializer; late initialization remains owned and cannot revive a timed-out submission. |

Each submitted Ray worker uses one attempt (`max_retries=0`,
`retry_exceptions=False`) and one runtime boundary (`max_calls=1`). Subprocess
workers likewise have one owned runtime boundary per accepted call. Subprocess
may carry an exact `WorldAllocation` grant with CPU IDs; Ray carries only verified
logical scheduler quantities and native evidence. Neither form creates a
cross-coordinator reservation or physical-isolation claim.

## Backends And Existing Environments

`dryml.execute.subprocess.SubProcessConfig` launches one owned local worker
group for each accepted call. `SubProcessFuture.process` is the borrowed launcher
until cleanup, and `.pid` is the worker-confirmed PID. The owned group is
reconciled on cleanup; descendants that escape the owned group remain caller
managed.

On POSIX, cleanup reaps an exited launcher before signalling its group. If
signalling races with exit, a fresh group-absence check can confirm cleanup;
launcher exit alone never proves that owned descendants have stopped. A group
that remains present or cannot be inspected still leaves cleanup incomplete.

```python
from pathlib import Path

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import CondaEnvironmentSpec
from dryml.execute import Executor
from dryml.execute.subprocess import SubProcessConfig

prefix = Path("/existing/conda-prefix")
config = SubProcessConfig(
    automatic_environment_discovery=False,
    environment_candidates=(CondaEnvironmentSpec(prefix=str(prefix)),),
)
with Executor(config) as executor:
    # The selected prefix must already contain compatible DRYML and dill.
    result = executor.run(sum, [4, 5], environment=EnvironmentRequirement())
```

`dryml.execute.ray.RayBackendConfig` attaches only to an existing trusted,
same-host, single-alive-node Ray deployment. It requires `ray[default]==2.56.0`
and compatible exact Python patch versions between coordinator and Ray worker.
The module import is lazy with respect to Ray; selecting/starting this backend
imports the SDK. It does not call a local cluster creation mode, initialize Ray
implicitly outside an executor start, or stop the caller-owned server.

```python
from dryml.execute import Executor
from dryml.execute.ray import RayBackendConfig
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement

one_cpu = WorldRequirement({
    "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
})
with Executor(RayBackendConfig(address="127.0.0.1:6379")) as executor:
    assert executor.run(sum, [8, 9], world=one_cpu) == 17
```

Backends report stable capability names for backend selection: `environment_selection`,
`world_admission`, `live_output`, and `running_cancellation`. The Ray backend also
reports `ray_existing_deployment`; its `world_admission` support is limited to one
role with logical CPU, memory, and GPU quantities. Subprocess supports the common
four capabilities for its owned local worker group.

`RayFuture.object_ref` is the native one-shot bootstrap task reference, not the
user result transport; use common `result()` for the callable value. Its
`server_address`, `worker_id`, `worker_pid`, and `node_id` are optional observed
native facts and clear after qualified cleanup. Existing Conda and direct venv
runtime selections must already contain compatible DRYML, dill, and Ray; Execute
does not create or modify either. A borrowed caller Ray driver remains connected
after executor close. Native cancellation and cleanup use best-effort SDK calls:
a cancellation request, task error, or resource delta alone is not a strict SDK
release acknowledgement.

Ray CPU, memory, and GPU values are logical scheduler quantities, not physical
isolation or device ownership. Subprocess and Ray backends can overlap physical
resources, and independent coordinators can overlap because there is no shared
cross-coordinator ledger. Each backend keeps only coordinator-and-backend scoped
charges. A missing qualified worker/task terminal association keeps a charge
unconfirmed and cleanup incomplete.

## Lifecycle And Limits

An accepted invocation receives a UUID submission ID, a coordinator-owned
invocation spool, and a reserved result allowance before launch. Spool budget
byte, file, and preflight counters are process-global across every active
executor lease, backend type, and spool parent. Their common quota tuple is held
while any lease is idle or cleanup has failed; only after the final lease reaches
zero may a new quota generation and tuple be configured. A live incompatible
tuple is a conflict, not a silently changed budget. Per-executor payload limits
and spool locations may still vary.
Invocation and result data remain in the submission child until qualified future
cleanup; the caller-owned spool parent, current directory, existing
environments, and Ray deployment are preserved.

The coordinator validates private protocol v4 and executes only after the worker
handshake/admission evidence and GO gate. Setup-bearing calls add `SETUP`,
setup-time `OUTPUT`, and `SETUP_READY` before `PAYLOAD`; `RESULT` is forbidden
before payload transfer. A setup `ERROR` cannot resume with readiness or payload,
though bounded final output frames are still drained. Callable graphs, callbacks,
native handles, and live futures are never transported as worker controls. The
result slot is reserved before launch so result receipt cannot bypass the quota.
Worker output is streamed through bounded frames into coordinator memory; this
implementation has no worker disk spool.

`ExecutionSnapshot.cleanup_scope` is `"owned-group"` for local subprocess and
`"worker"` for Ray. Cleanup releases only that scope. Escaped descendants,
caller-owned directories, a borrowed Ray driver, and the server itself are not
cleaned. `ExecutionUncertainError`, failed/cancelled outcomes, missing final
fences, mismatch, or incomplete cleanup are never presented as successful work.

For an exact pin, the fresh worker reports executable, prefix, base prefix, and
stable software evidence before `GO`, even if no software requirement was supplied.
A mismatch rejects before callable payload transfer. This is point-in-time
admission: external environment or package mutation after authorization is an
accepted limitation and is not monitored for the operation's lifetime.

## Testing And CI

Default tests do not provision external runtimes or Ray. Set
`DRYML_EXECUTE_INTEGRATION=1` only when all of these pre-existing test inputs
are available: `DRYML_TEST_RAY_ADDRESS` as `host:port`,
`DRYML_TEST_CONDA_PREFIX`, `DRYML_TEST_VENV_PYTHON`, and an absolute active
`CONDA_EXE`. Enabled collection fails when a prerequisite is absent or Ray is
not exactly 2.56.0; it does not skip.

The product never provisions a Ray server or Conda/venv target. The dedicated
Linux Python 3.12 CI job is different: its ephemeral runner deliberately creates
a job-owned Conda coordinator, same-interpreter venv, and `ray start --head`
fixture before pytest, installs the built DRYML artifact plus `dill` and pinned
Ray into both targets, exports the integration inputs, runs the real Ray and
existing-environment tests, then stops only that job-owned cluster. This is
developer test-fixture setup, not a runtime feature.

Dispatch can use this same existing-target API independently for declaration
probes and workloads. A Ray probe does not move a local or in-process workload,
and matching Ray configuration values still create separate probe and workload
owners. The dedicated CI fixture may exercise those paths; ordinary tests and
product calls require the caller to provide every target.

The lightweight Ubuntu/Windows Python 3.10-3.14 matrix remains framework-reduced
and exercises common, subprocess, package, and documentation coverage without a
mandatory Ray install. The heavy Ubuntu Python 3.10-3.13 matrix retains framework
coverage with the same Ray pin. Native Windows Ray/GPU support is not claimed.
The new enabled real-Ray fixture is verified on Linux Python 3.12 only until CI
results on a pushed revision are recorded.
