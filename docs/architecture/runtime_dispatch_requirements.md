# Runtime, Dispatch, and Requirements Architecture

## Status

Historical Sprint 0 baseline note, updated through Sprint 8 dispatch planning.

## Current State

DRYML currently separates declaration, planning, and execution across several modules:

- `dryml.annotations` stores sidecar metadata fragments on Python targets and owns the authoritative requirement collection/resolution APIs for live classes, functions, methods, and Definition/CDef method targets.
- `dryml.dispatch.Dispatcher.plan(...)` builds a `DispatchSpec`, `ExecutionRecipe`, and worker `ExecutionEnvelope`.
- `dryml.operations.OperationSpec` mappings are the canonical operation IR accepted by dispatch planning.
- `dryml.runtime.RuntimeMode` records the current process role.

## Problem Statement

Future sprints need to make normal Python-shaped dispatch calls work while preserving clear boundaries. Decorators should not execute work, dispatch kwargs should not silently erase hard requirements, and runtime enforcement should not be confused with process role.

## Guiding Principle

Decorators declare hard requirements and soft defaults. Dispatch kwargs select candidates. Candidate environment, world, and runtime selections are checked before launch; local allocation feasibility remains structural even when requirement policy relaxes compatibility reporting.

## Requirement Declaration vs Candidate Selection

`@dryml.env.req(...)` and `@dryml.world.req(...)` declare hard constraints. `@dryml.world.default(...)` and `@dryml.runtime.default(...)` declare overrideable defaults. Dispatch kwargs such as `environment=...`, `world=...`, and `runtime=...` should choose explicit candidates and override annotation defaults, not hard requirements.

For example, a target that requires a GPU must not silently become valid because a user passes a CPU-only world. A later `requirement_policy` may warn or ignore, but the default planning path should preserve the distinction.

## Current Dispatch Planning Behavior

At `a6d3550`, `Dispatcher.plan(...)` behaves as follows:

- `environment=None` is converted to `CurrentEnvironmentSpec().to_data()` and stored in dispatch/environment and the launch envelope.
- `world=None` is represented as `{"policy": "single_worker"}`.
- Mapping operations are validated as `OperationSpec`-like data and receive an operation ID.
- Python callables require `allow_pickle=True`; otherwise planning raises `DispatchPlanningError`.
- Python callables with `allow_pickle=True` use the existing small-pickle transport through `dryml.dispatch.operations:import_function` and are restricted to the same Python environment.
- Reporting includes requirement gather and merge steps, but dispatch does not yet resolve operation annotations into candidate checks.

## Requirement Collection Boundary

Sprint 3 adds `dryml.annotations.RequirementResolution` plus collection helpers such as `own_fragments`, `fragments_for_method`, `fragments_for_definition_method`, `resolve_target_requirements`, `resolve_method_requirements`, and `resolve_definition_method_requirements`. These APIs merge declared environment, world, and runtime requirements/defaults and preserve raw fragments, source traces, diagnostics, and report data.

Dispatch still does not consume those results during planning in Sprint 3. Candidate environment/world/runtime selection and compatibility checks remain deferred to later dispatch sprints.

## Dispatch Planning Pipeline

Dispatch normalizes user targets into an `OperationSpec`, collects code and annotation facts, merges hard requirements and defaults, selects candidate environment/world/runtime data, checks candidates, and then launches through the backend. Environment precedence is explicit, annotation default, context current, resolver, then current fallback; world precedence is explicit, annotation default, context current, synthesized local world, then fallback. DRYML-managed resolver probes are deadline-bounded; local inventory uses OS facts and any injected external command runner remains cooperative. An actual worker allocation is validated again before execution. `OperationSpec` remains the canonical internal IR even when normal users submit functions or CDef method calls directly.

The resolver consumes a bounded candidate prefix before probing, deduplicates
canonical identities, and validates worker probe protocol evidence before using
it. Planning metadata has bounded depth, item, string, and aggregate-node limits.
There is deliberately no cross-plan resolver, probe, or inventory cache.

## Runtime Enforcement Policy

`RuntimeMode` describes process role: `ORCHESTRATOR`, `PROBE`, `WORKER`, and `INLINE`. Runtime enforcement policy is separate and uses `RuntimeEnforcement.STRICT`, `RuntimeEnforcement.WARN`, and `RuntimeEnforcement.OFF`. Plain Python execution uses `RuntimeMode.INLINE` with a local runtime allocation view and `RuntimeEnforcement.OFF`; it does not add another runtime role.

Guard functions preserve prior behavior in `STRICT`. In `WARN`, DRYML enforcement guard violations emit `RuntimeWarning` where safe and continue. In `OFF`, those guard violations bypass safely without inventing resources. Python errors, import errors, serialization errors, and user code exceptions are not bypassed.

## Lightweight Code Probes

Sprint 5 adds `dryml.code.probe_target(...)`, `dryml.code.run_probe_request(...)`, `CodeProbeRequest`, and `CodeProbeResult`. A code probe runs the existing code analyzers under `RuntimeMode.PROBE` with `NoAllocation`, captures user-code stdout/stderr, and returns JSON-compatible code facts, diagnostics, and an optional `EnvironmentRecord` from the process that ran the probe.

The default lightweight analyzer set is `callables`, `source`, `symbol_capture`, and `direct_annotations`. Probe mode may import a target module to resolve an import path, so module-level import side effects remain possible. Probe mode does not intentionally execute target function bodies, instantiate target classes, run dynamic tracing, synthesize worlds, allocate workload resources, or change dispatch selection.

Current-process probes preserve the live `CodeTarget` wrapper for notebook/local functions, lambdas, and methods so local source/callable/annotation facts are not lost when a target has no stable import path. Subprocess probes require a serializable target reference, currently an import path or source spec, and return `code_probe.non_serializable_target` for live targets that cannot cross the JSON worker boundary.

Timeout enforcement is parent-side for subprocess probes. When a current-process probe receives a timeout for an import-path/source-backed target, it routes through the current Python executable worker so the timeout can be enforced. Live non-serializable current-process targets cannot be safely interrupted, so they return a structured `code_probe.timeout` diagnostic instead of pretending an in-process timeout is enforceable.

`python -m dryml.code.probe_worker --json` is the JSON worker protocol. It reads a schema-versioned serialized request from stdin and writes only serialized result JSON to stdout. Handled failures such as invalid JSON, unsupported schema versions, unknown algorithms, import failures, unsupported environments, and subprocess timeouts are represented as `DiagnosticFact` entries with `code_probe.*` diagnostic codes.

## Non-Goals

- This note does not specify exact implementation classes.
- Historical baseline sections do not supersede the current dispatch behavior
  described above.
- This note does not specify exact dispatch integration classes.
- Sprint 3 implements annotation requirement resolution APIs, but does not implement Python-shaped dispatch normalization or dispatch candidate checking.
- Sprint 4 implements runtime enforcement policy and current/default environment/world APIs, but still does not implement dispatch candidate checking.
- Sprint 5 implements lightweight code probes, but still does not implement dispatch requirement resolution or candidate checking.

## Source Anchors

- `src/dryml/dispatch/planner.py`
- `src/dryml/operations/specs.py`
- `src/dryml/annotations/decorators.py`
- `src/dryml/annotations/collect.py`
- `src/dryml/annotations/merge.py`
- `src/dryml/runtime/context.py`
- `src/dryml/runtime/enforcement.py`
- `src/dryml/runtime/modes.py`
- `src/dryml/code/probe.py`
- `src/dryml/code/probe_worker.py`

## Risks

Baseline tests can become brittle if they assert private plan internals. Sprint 0 tests should prefer plan, dispatch spec, recipe, envelope, and public exception behavior.

## Open Questions

- What public result shape should dispatch expose for requirement-resolution diagnostics?
- Should requirement-policy escape hatches live on dispatch kwargs, runtime policy, or both?
- Which operation metadata should persist requirement facts for execution records?

## Follow-Up Sprints

- Sprint 3: annotation collection and merge semantics.
- Sprint 5: lightweight code probe service.
- Sprint 6: Python-shaped operation normalization.
- Sprint 7: dispatch requirement resolution and candidate checking.
- Sprint 8: environment/world resolver behavior.
# Synthesized Worlds

Synthesis produces a `WorldSpec` only. Backend planning converts it into a
`WorldAllocation`, and only workers activate the allocation. This preserves the
orchestrator/notebook `NoAllocation` state while still allowing explain and
planning to inspect inventory and perform bounded environment probes.
