# Dispatch

`dryml.dispatch` coordinates immutable process-local workload configuration,
bounded declaration probing, and one explicit execution route. It is not a
backend pool, Store transport, resource scheduler, environment installer, Ray
provisioner, or Stage 8A retry/resume facility. There is no backend fallback and
no resource growth. Importing it, registering a backend configuration, or setting
defaults does not create a backend, inspect the host, open a Store, or reserve
resources.

## Configuration

Use `register_backend(name, config)` to retain an inert `BackendConfig`; a name
binds to its current configuration when passed to `with_options`, a default
setter, or `set_probe_default`. `backends()` returns a detached read-only map;
replacement requires `replace=True`, and `unregister_backend` only affects future
name resolution.

`BackendChoice` accepts an inert `BackendConfig`, `InProcess()`, or a registered
backend name. A name binds at the default, view, or probe-policy configuration
call that consumes it; later registry changes cannot retarget that choice, and an
unknown name raises `KeyError` there. The alias itself has no initialization,
discovery, reservation, or runtime-state side effects.

`set_worker_environment_default`, `set_worker_world_default`, and
`set_worker_python_default` update one process-local default. The first two are
hard worker requirements; `python` is an independent existing-environment
selector. Setting a worker world does not allocate resources or change the
caller's current Session allocation. `set_execute_backend_default(choice,
core=...)` captures one Execute configuration, registered name, or `InProcess()`
route; `None` clears it.

`with_options(env=..., world=..., python=..., backend=..., core=..., probe=...)`
returns an immutable view. Omitted values use the literal default `"inherit"`
and resolve lower-precedence views and current process defaults when an operation
starts. Explicit `None` clears that nullable selection. All `run`, `submit`, and
`explain` keyword arguments are workload data, including names such as `env` or
`backend`; options belong on the view. Every call snapshots one coherent effective
configuration before probing, so later setter changes cannot retarget it.

`InProcess()` is an explicit fieldless local route, not an Execute backend or
fallback. Its explanation reports `run` as its sole supported method and
`submit` rejects it before capture or probing. A missing route always raises an
explicit `ValueError` before capture or probing.

`explain(fn, *args, **kwargs)` validates synchronous root modality, snapshots
configuration, freezes an explicit worker selector, runs the independent bounded
probe, and may create the selected backend for bounded non-reserving discovery.
An in-process route instead checks fresh current-process evidence under a short
publication lease. It returns an immutable `DispatchReport`; it does not invoke
`fn`, serialize arguments, publish state, reserve resources, or issue an
admission ticket. Probe/configuration failures that can be represented safely
produce an ineligible report. `run` and `submit` expose the same failure as
`DispatchError(report)` before workload acceptance. Existing timeout, crash,
uncertain-delivery, and cleanup errors from owned probe/discovery lifecycles
continue unchanged.

## In-Process Execution

`with_options(backend=InProcess()).run(fn, *args, **kwargs)` probes first, then
obtains a publication-generation lease for final local admission. While that
lease is held, Dispatch derives the Session snapshot from that exact generation,
checks healthy non-orchestrator status, current software and exact Python-pin
evidence, and the selected process allocation against every hard requirement.
These checks are affirmative even when automatic Session enforcement axes are
disabled. A valued world requirement needs an exact selected allocation; host
inventory and a prior requested world are not allocation evidence.

The retained target guard is checked again under the lease before Dispatch calls
the original callable exactly once on the caller thread. Arguments, receiver,
direct mutations, return identity, and callable exceptions retain ordinary
Python behavior. The lease ends when that synchronous call returns or raises;
an iterator or awaitable returned by a synchronous root is data and is neither
advanced nor awaited. No Execute workload, core transport, Store export/save,
automatic refresh, reconfiguration, retry, rollback, or synthetic future is
created. `KeyboardInterrupt` and callable exceptions propagate unchanged.

Any explicit non-inherited `CoreOptions` field is worker-only and rejects the
local route. `core=None` and `CoreOptions()` with every field inherited are
inert. An incompatible publication attempted while the call is held receives
the runtime's existing `PublicationBusyError`; compatible framework status
finalization retains its existing publication behavior.

Reports contain only bounded diagnostic categories, redacted backend identifiers,
coverage and requirement results. They never retain call data, handles,
credentials, source, selectors, Stores, or reservations. Valid incomplete static
coverage is reported; `run` and `submit` emit `DispatchCoverageWarning`, while
`explain` does not warn.

## Probing And Static Coverage

`ProbeOptions` is an immutable, inert policy with defaults
`placement="auto"`, `execution_timeout=30.0`, `max_targets=256`, and
`max_depth=32`. `placement="auto"` uses an explicit probe backend when supplied;
otherwise it probes inline only with compatible current-process evidence and uses
an owned local subprocess when isolation is required. `placement="execute"`
uses the supplied backend or that local subprocess default. `placement="in_process"`
requires compatible current-process evidence and rejects a contradictory backend.
Probe placement is independent of the selected workload route. It never falls
back from an unavailable selected probe backend or changes the workload backend.

Both inline and Execute probes run the same `KernelCall` DAG:
`StaticDependenciesKernel`, `EnvironmentRequirementsKernel`, and
`WorldRequirementsKernel`. The analysis does not invoke the workload,
constructors, descriptors, dynamic hooks, or instance state. It follows direct
globals, closures, loaded-module paths, and one proven local alias. Wrapper bodies
are analyzed as written; `__wrapped__` and `__signature__` are not execution
authority. Receiver calls require a class-only non-shadowable proof, while
ordinary class construction can inspect known `__new__` and `__init__` bodies
without constructing an instance. Dynamic dispatch, callbacks, shadowable
receivers, opaque constructors, and exhausted bounds retain known declarations
but report incomplete coverage. Cycles terminate through target deduplication;
a cycle alone does not make otherwise resolved coverage incomplete.

The private request/result envelopes are version-local, nonpersistent, and capped
at 4 MiB. They retain at most 64 diagnostic entries with 512 characters per
field. Capture separately limits individual source reads to 1 MiB, aggregate
source reads to 8 MiB, candidate targets to 4,096, binding/call facts to 16,384,
and raw annotation occurrences to 4,096. These ceilings do not make static
analysis a full call-graph proof. A valid incomplete result warns and can proceed
when known requirements pass; malformed projections/results, conflicts, crashes,
timeouts, or cleanup failures stop submission rather than becoming empty results.

An `EnvironmentSpec` pin selects an existing interpreter, venv executable, or
Conda prefix/name exactly. Execute resolves and checks the worker executable,
prefix, base prefix, and stable software evidence before payload delivery. A pin
never becomes candidate search or a fallback; unsupported launcher forms and
unavailable selected targets fail. This is point-in-time evidence only: source
may already differ from loaded code, and an external environment can change after
admission. Observed preflight target drift and identity mismatch still fail.

## Backend Execution

For an Execute backend route, `submit` completes configuration, probing, target
validation, core payload preparation, and backend acceptance synchronously. It
returns the existing `CoreExecutionFuture`, so cancellation, uncertain outcome,
result recovery, and cleanup retain the core Execute contracts.

`run` performs the same one-off submission, waits for that future's recovered
result, and reconciles its owned cleanup. If both the workload result and
cleanup fail, the workload failure remains primary and the cleanup failure is
its cause. Dispatch does not retry work, adapt results, or choose another
backend after an unavailable or rejected selected backend.

The selected worker environment and world requirements are the combined
configured and discovered requirements. A resolved worker Python selector is
forwarded unchanged to Execute, which verifies it before payload delivery; it
never becomes candidate-search fallback. The retained probe guard is checked
before and during core preparation and acceptance. Observed target drift raises
`DispatchError` before workload acceptance, after reclaiming only preparation-
owned resources.

## Root Modality And Ownership

Dispatch accepts synchronous roots only. `explain`, `run`, and `submit` reject a
coroutine, generator, or async-generator root before capture, probing, backend
initialization, or direct invocation. A synchronous root may return an iterator,
generator object, coroutine object, or other awaitable as ordinary result data;
Dispatch never advances or awaits it, and a later caller-driven use is not covered
by the in-process admission lease. Backend-hosted results remain subject to the
existing Execute transport limits.

An Execute route returns the existing `CoreExecutionFuture` from `submit`; `run`
waits for its recovered result and preserves the established exception/cleanup
ordering. Probe and workload owners are distinct even when they use the same Ray
configuration. No report is an admission ticket or resource reservation. Dispatch
does not add automatic retries, result adaptation, cancellation semantics, resume,
or lost-response reconciliation.
