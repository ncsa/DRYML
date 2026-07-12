# ADR 0001: Code-Analysis Boundaries

## Status

Accepted for Sprint 9A static-analysis boundaries.

## Context

Current helpers for callable inspection, source extraction, AST inspection, method dispatch, and symbol/source references are spread across `dryml.code` and `dryml.core2.symbol`. Upcoming dispatch and probe work needs reusable code facts without embedding analysis in dispatch.

## Decision

`dryml.code` owns reusable analysis algorithms that discover facts about Python and DRYML code. `dryml.dispatch` consumes those facts and makes launch decisions. `dryml.annotations` owns fragment and merge semantics. `dryml.core2` owns stable semantic model primitives. `core2` must not depend on `dryml.code`; `dryml.code` may depend on `core2`.

Static analysis and future dynamic tracing share the `CodeAnalyzer`,
`CodeAnalysisContext`, `CodeAnalysisResult`, fact, and diagnostic protocol.
They are analysis modalities, not separate public analyzer hierarchies.
`analyze(...)` remains non-invoking. A future `trace(...)` API is the explicit
invocation-bearing path and is not exported until Sprint 9B implements it.

Probe execution selects an inline process, a managed subprocess, or a supported
Python environment for the same analyzer semantics. It does not create a second
analysis framework. A worker may run only analyzers installed and registered in
that worker environment.

## Consequences

Future analysis features should be implemented as reusable analyzers. Dispatch can stay focused on planning and launch decisions. Code probes can run the same algorithms in probe processes. Core primitives remain importable without higher-level analysis dependencies. Static call facts describe source-level possibilities only; dispatch does not consume them as hard requirements in Sprint 9A.

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

Sprint 9A adds bounded syntactic and conservative static-call analysis. Sprint 9B may implement the documented dynamic trace contract. Sprint 9C may add explicit dispatch policy for accepted facts and retire unchecked graph prototypes. Sprint 2 reviewed the `Method` model migration toward `core2.methods`.
