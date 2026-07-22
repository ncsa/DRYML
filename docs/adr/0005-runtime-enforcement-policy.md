# ADR 0005: Runtime Enforcement Policy

## Status

Proposed

## Context

`RuntimeMode` currently describes process role: orchestrator, probe, worker, or inline. Future work needs a way to run regular Python-like code or warning-only checks without inventing a new role.

## Decision

Runtime enforcement is separate from `RuntimeMode`. Future policy values should include `STRICT`, `WARN`, and `OFF`. `OFF` is not the same as a new runtime role; it changes how constraints are checked while the process still has a role and allocation state. Future regular-Python/plain mode should be expressed as enforcement policy plus normal runtime state, not as `RuntimeMode.PLAIN`.

## Consequences

Runtime mode remains stable. Dispatch and runtime guards can consult enforcement policy later. Users gain a clear path for exploratory/plain execution without redefining environment or world semantics.

## Alternatives Considered

Adding `PLAIN` as a runtime mode would mix role and enforcement. An environment-variable-only switch would be hard to inspect and test. Always-strict checks would be too rigid for migration and notebook workflows.

## Source Anchors

- `src/dryml/runtime/context.py`
- `src/dryml/runtime/modes.py`
- `src/dryml/runtime/guards.py`
- `src/dryml/dispatch/planner.py`

## Follow-up Work

Sprint 4 should add runtime enforcement policy and current env/world APIs. Sprint 7 should make dispatch respect enforcement and requirement policy.
