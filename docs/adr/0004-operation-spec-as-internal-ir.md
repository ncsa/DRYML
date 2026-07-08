# ADR 0004: OperationSpec As Internal IR

## Status

Proposed

## Context

`OperationSpec` is the current canonical serializable representation for operation calls. Normal users should eventually be able to submit Python-shaped targets without constructing specs manually.

## Decision

`OperationSpec` remains the canonical serializable internal IR. Normal users should not need to construct it for common calls. Future dispatch should accept `dispatch.submit(func, ...)`, `dispatch.submit(cdef, "train", ...)`, and advanced `dispatch.submit(operation_spec, ...)` paths by normalizing Python-shaped inputs into `OperationSpec` internally.

## Consequences

Existing advanced/internal spec flows remain stable. Dispatch gains one normalization layer instead of several unrelated launch APIs. Pickle remains an explicit same-Python convenience, not the default portable function path.

## Alternatives Considered

Removing `OperationSpec` would discard useful provenance and worker protocol structure. Requiring all users to construct specs would keep normal Python usage too low-level. Pickling all callables would be non-portable and would bypass import-path metadata.

## Source Anchors

- `src/dryml/operations/specs.py`
- `src/dryml/dispatch/planner.py`
- `src/dryml/dispatch/operations.py`

## Follow-up Work

Sprint 6 should implement Python-shaped dispatch operation normalization. Sprint 7 should perform requirement resolution around normalized operations.
