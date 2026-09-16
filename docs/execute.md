# Generic Execute

`dryml.execute` runs one trusted Python callable through an explicitly
selected backend. It is not `Repo`/`Store` transport, managed-operation
execution, Dispatch selection, a runtime/session API, or a safe-deserialization
boundary. Callable graphs and serialized payloads are trusted inputs.

There is no persistent resource ledger: each executor reports only
coordinator-and-backend scoped observations.

## Callable Transport

Execute supports function roots: importable functions, lambdas, nested
functions, closures with serializable captures, and importable unbound builtins.
It does not promise that every value satisfying `Callable` is transportable.
Stateful bound methods, bound builtin methods, and callable instances are
rejected synchronously before backend submission. Known DRYML core references
are also rejected by a fixed generic transport error; a DRYML core adapter is
deferred. The generic rejection does not expose type-specific core phrases or
the submitted value's private details.

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

`Executor.submit(fn, /, *args, kwargs=None, environment=None, world=None,
execution_timeout="inherit", stream_output=None, done_callbacks=(), output=None,
worker_setup=None)`
copies controls, serializes exactly one callable/argument graph to an
execution-owned spool child, and returns the backend's concrete
`ExecutionFuture`. `kwargs` is the workload keyword mapping; it is deliberately
separate from Execute controls. Invalid controls, requirements, preflight
serialization, spool capacity, or an executor closing fail synchronously.
Accepted asynchronous failures are published by the future.

`Executor.run` has the same keyword-only controls and returns `future.result()`.
It does not close the reusable executor. `Executor.with_options(...)` returns
an `ExecutorView` retaining the parent executor. A view has the same `run(fn,
/, *args, **kwargs)` and `submit(fn, /, *args, **kwargs)` workload boundary, so
control-named workload keywords remain ordinary workload data. The view never
owns a second backend, quota, future set, or close operation.

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

Module-level `submit(fn, /, *args, backend=..., ...)` and `run(...,
backend=..., ...)` require a `BackendConfig`; they retain a hidden owner until
the future's cleanup completes. One-off automatic cleanup uses the configured
bounded attempts and retry interval. A cleanup failure preserves the future and
its result/error for explicit recovery rather than reporting cleanup as complete.

`Executor.discover(environment=None, world=None, timeout=None)` returns a
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
its exception class. `ExecutionDeadlineExceeded` means owned termination after a
workload deadline was confirmed. `ExecutionUncertainError` means available
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

## Backends And Existing Environments

`dryml.execute.subprocess.SubProcessConfig` launches one owned local worker
group for each accepted call. `SubProcessFuture.process` is the borrowed launcher
until cleanup, and `.pid` is the worker-confirmed PID. The owned group is
reconciled on cleanup; descendants that escape the owned group remain caller
managed.

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

The coordinator validates private protocol v2 and executes only after the worker
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

The lightweight Ubuntu/Windows Python 3.10-3.14 matrix remains framework-reduced
and exercises common, subprocess, package, and documentation coverage without a
mandatory Ray install. The heavy Ubuntu Python 3.10-3.13 matrix retains framework
coverage with the same Ray pin. Native Windows Ray/GPU support is not claimed.
The new enabled real-Ray fixture is verified on Linux Python 3.12 only until CI
results on a pushed revision are recorded.
