# ADR 0001: Code-Analysis Boundaries

## Status

Accepted through Sprint 9C. ADR 0006 records the explicit maintained-full
verification deviation accepted for Sprint 9C closeout.

## Context

Current helpers for callable inspection, source extraction, AST inspection, method dispatch, and symbol/source references are spread across `dryml.code` and `dryml.core2.symbol`. Upcoming dispatch and probe work needs reusable code facts without embedding analysis in dispatch.

## Decision

`dryml.code` owns reusable analysis algorithms that discover facts about Python and DRYML code. `dryml.dispatch` consumes those facts and makes launch decisions. `dryml.annotations` owns fragment and merge semantics. `dryml.core2` owns stable semantic model primitives. `core2` must not depend on `dryml.code`; `dryml.code` may depend on `core2`.

Static analysis and dynamic tracing share the `CodeAnalyzer`,
`CodeAnalysisContext`, `CodeAnalysisResult`, fact, and diagnostic protocol.
They are analysis modalities, not separate public analyzer hierarchies.
`analyze(...)` remains non-invoking. `trace(...)` is the explicit
invocation-bearing path. It requires dynamic-execution permission and invokes a
supported trusted synchronous function once with bounded Definition/CDef proxies
in the current process.

Probe execution selects an inline process, a managed subprocess, or a supported
Python environment for the same analyzer semantics. It does not create a second
analysis framework. A worker may run only analyzers installed and registered in
that worker environment.

## Consequences

Future analysis features should be implemented as reusable analyzers. Dispatch can stay focused on planning and launch decisions. Code probes can run the same algorithms in probe processes. Core primitives remain importable without higher-level analysis dependencies. Static call facts describe source-level possibilities only; dispatch does not consume them as hard requirements in Sprint 9A.

`dynamic_trace` is registered for protocol consistency but ordinary analyzer and
probe invocation returns a requires-trace-facade diagnostic and never invokes the
target. It is absent from both default analyzer tuples. Dynamic call facts are
observations from one explicit inline run, not exact cross-run Definition
identity. Dispatch consumes them only after an explicit, strict
`analysis_policy.dynamic_trace` request, validates bounded result evidence, and
passes direct plus accepted annotation fragments to `dryml.annotations` for the
authoritative merge. Tracing remains cooperative trusted-code execution, not a
sandbox, subprocess, selected-environment facility, or hard-timeout boundary.
Dispatch gates opt-in targets as exact synchronous Python functions before
generic callable inspection or pickle creation. Its versioned diagnostic carrier
admits only the exact 9B no-summary pre-execution diagnostic set; stale or
malformed evidence is rejected after independent bounded summary/call validation
and never becomes structural input. Unknown transport tokens and provenance
overflow fail closed, and carrier redaction excludes exception/source/environment
data and live objects.

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

Sprint 9A added bounded syntactic and conservative static-call analysis. Sprint
9B added the bounded current-process dynamic trace facade. Sprint 9C added the
explicit dispatch policy and bounded planning carrier for accepted facts. Its
maintained-full closeout exception and required future rerun conditions are
recorded in ADR 0006. Unchecked graph prototypes remain absent from tracked
distributions. Sprint 2 reviewed the `Method` model migration toward
`core2.methods`.
