# ADR 0001: Code-Analysis Boundaries

## Status

Proposed

## Context

Current helpers for callable inspection, source extraction, AST inspection, method dispatch, and symbol/source references are spread across `dryml.code` and `dryml.core2.symbol`. Upcoming dispatch and probe work needs reusable code facts without embedding analysis in dispatch.

## Decision

`dryml.code` owns reusable analysis algorithms that discover facts about Python and DRYML code. `dryml.dispatch` consumes those facts and makes launch decisions. `dryml.annotations` owns fragment and merge semantics. `dryml.core2` owns stable semantic model primitives. `core2` must not depend on `dryml.code`; `dryml.code` may depend on `core2`.

## Consequences

Future analysis features should be implemented as reusable analyzers. Dispatch can stay focused on planning and launch decisions. Code probes can run the same algorithms in probe processes. Core primitives remain importable without higher-level analysis dependencies.

## Alternatives Considered

Putting analysis inside dispatch would duplicate algorithms and tie them to one planning path. Putting analysis inside annotations would blur metadata collection with code inspection. Leaving unchecked `dryml.graph` or scattered helpers separate would make later dispatch behavior hard to test and review.

## Source Anchors

- `src/dryml/code/callable_info.py`
- `src/dryml/code/source.py`
- `src/dryml/code/ast_tools.py`
- `src/dryml/code/method.py`
- `src/dryml/code/traits.py`
- `src/dryml/core2/symbol.py`

## Follow-up Work

Sprint 1 should add the fact-oriented analysis API. Sprint 9 should consider optional dynamic tracing. Sprint 2 should review whether `Method`, method handles, `Traits`, and `CompilerInfo` move toward `core2.methods`.
