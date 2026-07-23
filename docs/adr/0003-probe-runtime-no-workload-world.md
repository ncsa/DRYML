# ADR 0003: Probe Runtime Without Workload World

## Status

Proposed

## Context

Code and environment probes need to inspect metadata before final workload launch. They should be lightweight and avoid consuming workload resources such as GPUs.

## Decision

Probes run in `RuntimeMode.PROBE`, should use `NoAllocation`, and do not require the final workload world. Probe jobs are metadata, code, and environment inspection jobs, not workload workers.

## Consequences

Code probes can run as lightweight single-process jobs. Probe results should be facts and diagnostics. User imports may still have side effects, so probe execution must remain explicit and isolated where possible.

## Alternatives Considered

Running probes inside the final worker allocation would waste resources and delay validation. Running probes only in the orchestrator may import heavy or unsafe user code in the control process. Skipping probes and relying on worker failures would produce late, low-quality errors.

## Source Anchors

- `src/dryml/runtime/context.py`
- `src/dryml/runtime/modes.py`
- `src/dryml/environments/probe.py`
- `src/dryml/environments/introspection.py`

## Follow-up Work

Sprint 5 should add lightweight code probe behavior. Sprint 7 should let dispatch consume probe results when needed.
