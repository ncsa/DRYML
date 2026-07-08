# ADR 0002: Dispatch Requirement Resolution

## Status

Proposed

## Context

Decorators can already attach environment, world, and runtime fragments. Dispatch can already accept explicit `environment`, `world`, and `runtime` kwargs. Later sprints need a rule for how these interact.

## Decision

Hard requirements constrain candidate environments, worlds, and runtime choices. Dispatch kwargs select explicit candidates. Dispatch kwargs override annotation defaults, not hard requirements.

For example, a function that declares a GPU world requirement should not become valid because a caller passes a CPU-only world. A future `requirement_policy="warn"` or `requirement_policy="ignore"` may provide an explicit escape hatch, but silent erasure is not the default.

## Consequences

Requirement collection and candidate compatibility checks become part of dispatch planning. Diagnostics must explain whether a failure came from a hard requirement, a default, an explicit kwarg, or provider/probe facts.

## Alternatives Considered

Explicit kwargs could replace all requirements, but that would make decorators unreliable. Annotations could be advisory only, but then dispatch could not protect workload placement. No warn/ignore escape hatch would be too strict for exploratory use and migration.

## Source Anchors

- `src/dryml/annotations/merge.py`
- `src/dryml/dispatch/planner.py`
- `src/dryml/worlds/compatibility.py`
- `src/dryml/environments/requirements.py`

## Follow-up Work

Sprint 3 should refine requirement collection semantics. Sprint 7 should implement dispatch requirement resolution and candidate checks.
