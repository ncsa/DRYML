# Runtime, Dispatch, and Requirements Architecture

## Status

Current architecture note.

## Current State

DRYML currently separates declaration, planning, and execution across several modules:

- `dryml.annotations` stores sidecar metadata fragments on Python targets and owns the authoritative requirement collection/resolution APIs for live classes, functions, methods, and Definition/CDef method targets.
- `dryml.dispatch.Dispatcher.plan(...)` builds a `DispatchSpec`, `ExecutionRecipe`, and worker `ExecutionEnvelope`.
- `dryml.operations.OperationSpec` mappings are the canonical operation IR accepted by dispatch planning.
- `dryml.runtime.RuntimeMode` records the current process role.
- `dryml.code.trace(...)` is the sole invocation-bearing analysis facade;
  `dryml.dispatch` owns only explicit opt-in policy, worker-effective invocation
  reconstruction, provenance admission, and launch decisions.

## Ownership Boundary

Normal Python-shaped dispatch is shipped while preserving clear boundaries.
Decorators do not execute work, dispatch kwargs do not silently erase hard
requirements, and runtime enforcement is distinct from process role.

## Guiding Principle

Decorators declare hard requirements and soft defaults. Dispatch kwargs select candidates. Candidate environment, world, and runtime selections are checked before launch; local allocation feasibility remains structural even when requirement policy relaxes compatibility reporting.

## Requirement Declaration vs Candidate Selection

`@dryml.env.req(...)` and `@dryml.world.req(...)` declare hard constraints. `@dryml.world.default(...)` and `@dryml.runtime.default(...)` declare overrideable defaults. Dispatch kwargs such as `environment=...`, `world=...`, and `runtime=...` should choose explicit candidates and override annotation defaults, not hard requirements.

For example, a target that requires a GPU must not silently become valid because a user passes a CPU-only world. `requirement_policy="warn"` or `"ignore"` can relax compatibility enforcement, but the default planning path preserves the distinction.

## Current Dispatch Planning Behavior

`Dispatcher.plan(...)` behaves as follows:

- `environment=None` is converted to `CurrentEnvironmentSpec().to_data()` and stored in dispatch/environment and the launch envelope.
- An importable Python function uses an import-path operation; explicit or
  non-importable pickle transport is `pickle_small` and remains current-Python
  only.
- Mapping operations are validated as `OperationSpec`-like data and receive an
  operation ID; planning metadata and per-run trace evidence are not operation
  metadata.
- Planning resolves annotation requirements/defaults, selects candidates, checks
  compatibility, and validates backend allocation before launch.

## Requirement Collection Boundary

`dryml.annotations.RequirementResolution` and its collection helpers merge
declared environment, world, and runtime requirements/defaults while preserving
raw fragments, source traces, diagnostics, and report data. Dispatch consumes
that resolution during planning but does not reimplement its merge semantics.

## Dispatch Planning Pipeline

Dispatch normalizes user targets into an `OperationSpec`, collects code and annotation facts, merges hard requirements and defaults, selects candidate environment/world/runtime data, checks candidates, and then launches through the backend. Environment precedence is explicit, annotation default, context current, resolver, then current fallback; world precedence is explicit, annotation default, context current, synthesized local world, then fallback. DRYML-managed resolver probes are deadline-bounded; local inventory uses OS facts and any injected external command runner remains cooperative. An actual worker allocation is validated again before execution. `OperationSpec` remains the canonical internal IR even when normal users submit functions or CDef method calls directly.

An `analysis_policy={"dynamic_trace": True | DynamicTracePolicy(...)}` request
adds one narrow branch. It is default-off even if a supplied
`CodeAnalysisContext` permits dynamic execution. Dispatch derives a fresh,
forced-collect trace context, reconstructs the resolver-equivalent current
worker invocation without building CDefs, and invokes one eligible live function
in the orchestrator process. Direct fragments precede admitted trace fragments;
annotations remains the sole merge authority. Trace failure, incomplete or
rejected evidence, and provenance overflow are structural planning failures,
independent of strict/warn/ignore. This trusted-code execution has neither a
sandbox nor hard timeout and is never relocated to a probe or worker.
At its public policy boundary, dispatch creates a private deep JSON metadata
snapshot of the caller context. Discovery, trace-input identity, and facade
derivation use that snapshot only, so a later mutation cannot change an
in-flight request's effective metadata.

The dispatch trace projection is versioned and bounded. It carries per-request
input/run identity and recognized `pre_execution_failed`/`evidence_rejected`
states in dispatch, recipe, envelope, and explanation metadata only. Immutable
OperationSpec metadata is trace-free; reserved planning keys are removed before
operation-sidecar publication.

Admission is fail-closed but diagnostic-preserving. A no-summary pre-execution
carrier is accepted only for the exact documented diagnostic-only/no-fact outcome set
and known non-start; stale envelope/target evidence, malformed summaries,
unknown outcomes, and mixed evidence are `evidence_rejected`. Summary/call wires
are independently validated before envelope rejection, so validated evidence
can prove `execution_started=true`; otherwise the rejected carrier represents
unknown start as `null`. Retained rejected evidence is diagnostic-only and never
enters annotations resolution. The carrier accepts only `import_path`,
`pickle_small`, `operation_spec`, and `method_call` transport tokens, rejects
unknown tokens, enforces `max_calls` 1..10,000 plus dispatch count/depth/
string/byte limits, and excludes exception text, tracebacks, raw source,
environment values, streams, live objects, and arbitrary repr. Its `calls` field
uses the specified ordered full `DynamicCallFact` wires; accepted annotation
fragments enter canonical deduplication and authoritative resolution unchanged.
Because v1 has no redacted alternate call schema, dispatch fails closed before
resolution and persistence when a call has recorded arguments, an annotation has
a local source path or unrecognized source/target metadata, or a fragment carries
environment overrides. The established bounded
`legacy_environment_fragment_mode` source metadata remains semantic and is
preserved unchanged. Projection overflow retains a valid summary with empty
calls as `provenance_limit_exceeded`; it never truncates or rewrites evidence.
For `pickle_small`, a final candidate that is not the current Python after an
accepted trace is non-launchable but preserves that completed carrier through
cleanup.

The resolver consumes a bounded candidate prefix before probing, deduplicates
canonical identities, and validates worker probe protocol evidence before using
it. Planning metadata has bounded depth, item, string, and aggregate-node limits.
There is deliberately no cross-plan resolver, probe, or inventory cache.

## Runtime Enforcement Policy

`RuntimeMode` describes process role: `ORCHESTRATOR`, `PROBE`, `WORKER`, and `INLINE`. Runtime enforcement policy is separate and uses `RuntimeEnforcement.STRICT`, `RuntimeEnforcement.WARN`, and `RuntimeEnforcement.OFF`. Plain Python execution uses `RuntimeMode.INLINE` with a local runtime allocation view and `RuntimeEnforcement.OFF`; it does not add another runtime role.

Guard functions preserve prior behavior in `STRICT`. In `WARN`, DRYML enforcement guard violations emit `RuntimeWarning` where safe and continue. In `OFF`, those guard violations bypass safely without inventing resources. Python errors, import errors, serialization errors, and user code exceptions are not bypassed.

## Lightweight Code Probes

`dryml.code.probe_target(...)`, `dryml.code.run_probe_request(...)`,
`CodeProbeRequest`, and `CodeProbeResult` run existing code analyzers under
`RuntimeMode.PROBE` with `NoAllocation`, capture user-code stdout/stderr, and
return JSON-compatible code facts, diagnostics, and an optional
`EnvironmentRecord` from the process that ran the probe.

The default lightweight analyzer set is `callables`, `source`, `symbol_capture`, and `direct_annotations`. Probe mode may import a target module to resolve an import path, so module-level import side effects remain possible. Probe mode does not intentionally execute target function bodies, instantiate target classes, run dynamic tracing, synthesize worlds, allocate workload resources, or change dispatch selection.

Current-process probes preserve the live `CodeTarget` wrapper for notebook/local functions, lambdas, and methods so local source/callable/annotation facts are not lost when a target has no stable import path. Subprocess probes require a stable import path. `CodeTargetSpec.source_spec` remains serializable descriptive data, but source-spec reconstruction is not implemented and worker requests return `code_probe.source_spec_reconstruction_unavailable` rather than attempting to fabricate a live target.

Timeout enforcement is parent-side for subprocess probes. When a current-process probe receives a timeout for an import-path target, it routes through the current Python executable worker so the timeout can be enforced. Live non-serializable current-process targets cannot be safely interrupted, so they return a structured `code_probe.timeout` diagnostic instead of pretending an in-process timeout is enforceable.

`python -m dryml.code.probe_worker --json` is the JSON worker protocol. It reads a schema-versioned serialized request from stdin and writes only serialized result JSON to stdout. Handled failures such as invalid JSON, unsupported schema versions, unknown algorithms, import failures, unsupported environments, and subprocess timeouts are represented as `DiagnosticFact` entries with `code_probe.*` diagnostic codes.

## Non-Goals

- This note does not specify exact implementation classes.
- This note does not specify exact dispatch integration classes.

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

Tests can become brittle if they assert private plan internals. Prefer plan,
dispatch spec, recipe, envelope, and public exception behavior.

## Resolved Decisions

- `DispatchExplanation` and `DispatchPlanningResolution` expose bounded structured
  requirement, selection, check, resolver, synthesis, and allocation diagnostics.
- Requirement policy is a dispatch planning input; runtime enforcement remains a
  distinct worker-runtime policy.
- DRYML-owned planning metadata persists canonical requirement facts and bounded
  decisions without probe streams, environment secrets, or live objects.

# Synthesized Worlds

Synthesis produces a `WorldSpec` only. Backend planning converts it into a
`WorldAllocation`, and only workers activate the allocation. This preserves the
orchestrator/notebook `NoAllocation` state while still allowing explain and
planning to inspect inventory and perform bounded environment probes.
