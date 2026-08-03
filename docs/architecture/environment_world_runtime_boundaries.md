# Environment, World, and Runtime Boundaries

## Status

Current ownership boundary for environments, worlds, runtime, and dispatch.

## Current State

DRYML has distinct modules for software environments, resource worlds, and process-local runtime state. `dryml.session` is a persistent facade over those boundaries, not a replacement authority. Dispatch records a requested world separately from its actual allocation, resolves explicit environment registries only after higher-precedence selections are absent, and can synthesize a bounded local world from a hard requirement. Context-local defaults remain available through `dryml.environments.current()` and `dryml.worlds.current()`.

## Definitions

`environment` means Python, Conda, container, package, and software metadata or requirements.

`world` means topology, resources, allocation request, or actual allocation.

`runtime` means current process role, active allocation view, runtime spec, and enforcement policy.

`allocation` means actual resources assigned to a process from a world allocation.

`enforcement` means how strictly DRYML checks environment/world/runtime constraints; it is not the same as runtime role. Requirement axes select compatibility only and never weaken lifecycle, allocation, materialization, or visibility guards.

`probe` means a lightweight process role for inspection; it is not a workload worker and is not tied to the final workload world.

## Environment vs World vs Runtime

Environment probing and world allocation are separate concerns. A probe can inspect Python/package metadata without knowing the final GPU topology. A world spec can request CPUs, memory, accelerators, roles, and replicas without changing current Python packages. Runtime activation describes what the current process is allowed to see and do.

## Probe Runtime Direction

Probe processes do not need the final workload world and should normally run without GPU allocation. They should use `RuntimeMode.PROBE` and `NoAllocation`, producing facts and diagnostics rather than user workload results.

`dryml.code.probe_target(...)` is the combined code-analysis probe surface. It can run in the current Python process, through `CurrentEnvironmentSpec`, or through an explicit Python executable worker. It may optionally include an `EnvironmentRecord` collected inside the process that ran the analysis. Unsupported environment launch paths return diagnostics rather than creating environments, solving packages, using containers, or synthesizing worlds.

Current-process probes can inspect live local/notebook targets because they preserve the live `CodeTarget` wrapper. Worker/subprocess probes cross a JSON boundary and therefore require a stable `module:qualname` import path; `source_spec` remains descriptive data and is not reconstructed. Timeout enforcement is subprocess-based; current-process probes with timeouts route import-path targets through the current Python worker and reject other targets with a structured diagnostic.

## Notebook Current/Default State

Fresh `dryml.session` state is intentionally `RuntimeMode.NONE` unchecked Python
with `NoAllocation` and `OFF`. A managed session
creates the current-process allowance and applies pre-import visibility controls;
an orchestrator session has checked control-plane behavior with no current
allocation. `session.worker_env_request(...)` and
`session.worker_world_request(...)` remain separate default worker candidates, so
a CPU-only managed notebook can dispatch a GPU worker without exposing it
locally. Environment requirements are software compatibility only. Process memory and
accelerator allocator memory are distinct, and both remain declarative unless a
specific process or adapter control reports a stronger per-control status.

`dryml.environments.current()` and `dryml.worlds.current()` represent context-local notebook/session defaults for future dispatches. In contrast, `dryml.runtime.active_runtime().allocation` means the actual allocation of this process. Setting a current world does not allocate resources and does not imply that the current process owns that world.

`dryml.worlds.discover_current()` first returns the explicit context-local current
world. When no explicit current world is set, it returns the caller default rather
than synthesizing worlds or converting runtime allocation into a requested world.

## Dispatch Candidate Selection Direction

Dispatch combines current/default state, explicit kwargs, annotation defaults, and hard requirements into candidate selections. Explicit candidates override defaults but remain checked against hard requirements; strict compatibility on all axes is the dispatch default, while advanced policies and masks affect compatibility only, never structural launchability, allocation feasibility, or target importability.

## Notebook and Orchestration Examples

In a notebook, a user may set default requested environment/world candidates for later dispatch while the notebook process remains orchestrator mode with no workload allocation. During local worker execution, a child process validates complete selections and enters worker mode with a real allocation before handshake or workload setup. During a code probe, a child process can enter probe mode with no workload allocation and inspect metadata safely.

## Control-Plane Boundary

Strict orchestration is a trusted-code lifecycle boundary, not a sandbox. It
publishes a session-wide definition mode and accelerator-hidden `NoAllocation`
parent. Definition, concrete, selector, and space work may proceed after setup;
new live Object materialization and local managed workload execution must move to
a managed process or explicit worker. A future worker request may still select
accelerators without changing parent visibility.

## Non-Goals

- Current/default APIs do not allocate resources.
- `worlds.discover_current()` does not synthesize worlds.
- Code probes do not execute target function bodies, dynamic tracing, workload
  workers, or dispatch candidate selection.
- Environment registry resolution is bounded candidate selection, not package
  solving.

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

## Resolved Decisions

- Environment requirements are resolved against bounded `EnvironmentRecord`
  facts produced by `dryml.environments` probes; labels remain prefilters only.
- `dryml.environments` owns probe mechanics and reports, while dispatch decides
  when precedence permits probing and reuses the selected evidence.

# Local Discovery Boundary

`dryml.worlds` owns import-safe local inventory and deterministic requested-world
synthesis. `dryml.environments` owns explicit registry candidate resolution and
probing. `dryml.dispatch` applies precedence, validates the selected candidates,
and asks a backend to create allocations; it does not make registry state global.

An advanced notebook can keep its registry in a local variable and use
`environments.use(...)`/`worlds.use(...)` for temporary defaults. Inventory and
synthesis still leave that low-level notebook path at `NoAllocation`; only a
worker receives the actual allocation. Resolver and inventory work are bounded
per request and do not use cross-plan caching.
