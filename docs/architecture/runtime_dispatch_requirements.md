# Runtime, Dispatch, and Requirements Architecture

## Status

Proposed Sprint 0 baseline note. Current-state claims are anchored to baseline commit `a6d3550` and are intended to document behavior, not change it.

## Current State

DRYML currently separates declaration, planning, and execution across several modules:

- `dryml.annotations` stores sidecar metadata fragments on Python targets and owns the authoritative requirement collection/resolution APIs for live classes, functions, methods, and Definition/CDef method targets.
- `dryml.dispatch.Dispatcher.plan(...)` builds a `DispatchSpec`, `ExecutionRecipe`, and worker `ExecutionEnvelope`.
- `dryml.operations.OperationSpec` mappings are the canonical operation IR accepted by dispatch planning.
- `dryml.runtime.RuntimeMode` records the current process role.

## Problem Statement

Future sprints need to make normal Python-shaped dispatch calls work while preserving clear boundaries. Decorators should not execute work, dispatch kwargs should not silently erase hard requirements, and runtime enforcement should not be confused with process role.

## Guiding Principle

Decorators declare hard requirements and soft defaults. Dispatch kwargs select candidates. Candidate environment, world, and runtime selections should eventually be checked against hard requirements before launch.

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

## Future Dispatch Planning Pipeline

A later implementation should normalize user targets into an `OperationSpec`, collect code and annotation facts, merge hard requirements and defaults, select candidate environment/world/runtime data, check candidates against hard requirements, then launch through the backend. `OperationSpec` remains the canonical internal IR even when normal users submit functions or CDef method calls directly.

## Runtime Enforcement Policy

`RuntimeMode` describes process role: `ORCHESTRATOR`, `PROBE`, `WORKER`, and `INLINE`. Runtime enforcement policy is separate and uses `RuntimeEnforcement.STRICT`, `RuntimeEnforcement.WARN`, and `RuntimeEnforcement.OFF`. Plain Python execution uses `RuntimeMode.INLINE` with a local runtime allocation view and `RuntimeEnforcement.OFF`; it does not add another runtime role.

Guard functions preserve prior behavior in `STRICT`. In `WARN`, DRYML enforcement guard violations emit `RuntimeWarning` where safe and continue. In `OFF`, those guard violations bypass safely without inventing resources. Python errors, import errors, serialization errors, and user code exceptions are not bypassed.

## Non-Goals

- This note does not specify exact implementation classes.
- This note does not change dispatch behavior.
- This note does not specify exact dispatch integration classes.
- Sprint 3 implements annotation requirement resolution APIs, but does not implement Python-shaped dispatch normalization or dispatch candidate checking.
- Sprint 4 implements runtime enforcement policy and current/default environment/world APIs, but still does not implement dispatch candidate checking.

## Source Anchors

- `src/dryml/dispatch/planner.py`
- `src/dryml/operations/specs.py`
- `src/dryml/annotations/decorators.py`
- `src/dryml/annotations/collect.py`
- `src/dryml/annotations/merge.py`
- `src/dryml/runtime/context.py`
- `src/dryml/runtime/enforcement.py`
- `src/dryml/runtime/modes.py`

## Risks

Baseline tests can become brittle if they assert private plan internals. Sprint 0 tests should prefer plan, dispatch spec, recipe, envelope, and public exception behavior.

## Open Questions

- What public result shape should dispatch expose for requirement-resolution diagnostics?
- Should requirement-policy escape hatches live on dispatch kwargs, runtime policy, or both?
- Which operation metadata should persist requirement facts for execution records?

## Follow-Up Sprints

- Sprint 3: annotation collection and merge semantics.
- Sprint 6: Python-shaped operation normalization.
- Sprint 7: dispatch requirement resolution and candidate checking.
- Sprint 8: environment/world resolver behavior.
