# Environment, World, and Runtime Boundaries

## Status

Historical baseline note anchored to `a6d3550`, updated through Sprint 8 local
inventory, world synthesis, and environment resolution.

## Current State

DRYML has distinct modules for software environments, resource worlds, and process-local runtime state. Dispatch records a requested world separately from its actual allocation, resolves explicit environment registries only after higher-precedence selections are absent, and can synthesize a bounded local world from a hard requirement. Notebook/session defaults are context-local through `dryml.environments.current()` and `dryml.worlds.current()`.

## Definitions

`environment` means Python, Conda, container, package, and software metadata or requirements.

`world` means topology, resources, allocation request, or actual allocation.

`runtime` means current process role, active allocation view, runtime spec, and enforcement policy.

`allocation` means actual resources assigned to a process from a world allocation.

`enforcement` means how strictly DRYML checks environment/world/runtime constraints; it is not the same as runtime role.

`probe` means a lightweight process role for inspection; it is not a workload worker and is not tied to the final workload world.

## Environment vs World vs Runtime

Environment probing and world allocation are separate concerns. A probe can inspect Python/package metadata without knowing the final GPU topology. A world spec can request CPUs, memory, accelerators, roles, and replicas without changing current Python packages. Runtime activation describes what the current process is allowed to see and do.

## Probe Runtime Direction

Probe processes do not need the final workload world and should normally run without GPU allocation. They should use `RuntimeMode.PROBE` and `NoAllocation`, producing facts and diagnostics rather than user workload results.

`dryml.code.probe_target(...)` is the combined code-analysis probe surface. It can run in the current Python process, through `CurrentEnvironmentSpec`, or through an explicit Python executable worker. It may optionally include an `EnvironmentRecord` collected inside the process that ran the analysis. Unsupported environment launch paths return diagnostics rather than creating environments, solving packages, using containers, or synthesizing worlds.

Current-process probes can inspect live local/notebook targets because they preserve the live `CodeTarget` wrapper. Worker/subprocess probes cross a JSON boundary and therefore require a serializable target reference such as an import path or source spec. Timeout enforcement is subprocess-based; current-process probes with timeouts route serializable targets through the current Python worker and reject live non-serializable targets with a diagnostic.

## Notebook Current/Default State

`dryml.environments.current()` and `dryml.worlds.current()` represent context-local notebook/session defaults for future dispatches. In contrast, `dryml.runtime.active_runtime().allocation` means the actual allocation of this process. Setting a current world does not allocate resources and does not imply that the current process owns that world.

`dryml.worlds.discover_current()` first returns the explicit context-local current world. In Sprint 4, when no explicit current world is set, it returns the caller default rather than synthesizing worlds or converting runtime allocation into a requested world.

## Dispatch Candidate Selection Direction

Dispatch combines current/default state, explicit kwargs, annotation defaults, and hard requirements into candidate selections. Explicit candidates override defaults but remain checked against hard requirements; `warn` and `ignore` may relax compatibility reporting but never structural launchability, allocation feasibility, or target importability.

## Notebook and Orchestration Examples

In a notebook, a user may set a default requested world for later dispatch while the notebook process remains orchestrator mode with no workload allocation. During local worker execution, a child process enters worker mode with a real CPU-only allocation. During a code probe, a child process can enter probe mode with no workload allocation and inspect metadata safely.

## Non-Goals

- Sprint 4 current/default APIs do not allocate resources.
- Sprint 4 `worlds.discover_current()` does not synthesize worlds.
- Sprint 4 runtime enforcement does not implement dispatch candidate checking.
- Sprint 5 code probes do not execute target function bodies, dynamic tracing, workload workers, or dispatch candidate selection.
- Sprint 0 does not implement environment or world registry/resolver behavior.

## Source Anchors

- `src/dryml/environments/__init__.py`
- `src/dryml/environments/current.py`
- `src/dryml/environments/probe.py`
- `src/dryml/environments/introspection.py`
- `src/dryml/worlds/__init__.py`
- `src/dryml/worlds/current.py`
- `src/dryml/worlds/compatibility.py`
- `src/dryml/runtime/context.py`
- `src/dryml/runtime/modes.py`
- `src/dryml/code/probe.py`
- `src/dryml/code/probe_worker.py`

## Open Questions

- What minimum environment facts are required before dispatch can resolve requirements?
- Should probe scheduling be owned by dispatch or by a provider/probe service?

## Follow-Up Sprints

- Sprint 4: runtime enforcement and current env/world APIs.
- Sprint 5: lightweight code probe service.
- Sprint 7: dispatch requirement checks.
- Sprint 8: resolver and registry behavior.
# Sprint 8 Local Discovery Boundary

`dryml.worlds` owns import-safe local inventory and deterministic requested-world
synthesis. `dryml.environments` owns explicit registry candidate resolution and
probing. `dryml.dispatch` applies precedence, validates the selected candidates,
and asks a backend to create allocations; it does not make registry state global.

An ordinary notebook keeps its registry in a local variable and uses
`environments.use(...)`/`worlds.use(...)` for restorable defaults. Inventory and
synthesis still leave the notebook at `NoAllocation`; only a worker receives the
actual allocation. Resolver and inventory work are bounded per request, with
cross-plan caching deferred to a later performance sprint.
