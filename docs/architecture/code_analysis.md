# Code Analysis Architecture

## Status

Sprint 9A implementation note for the reusable `dryml.code` analysis API.

## Current State

`dryml.code` contains a fact-oriented analysis layer plus compatibility helper modules for callable inspection, source extraction, and AST access collection. `dryml.core2.symbol` owns import references and source-backed symbol references. `dryml.core2.methods` owns method semantic objects such as `Method`, `Traits`, `CompilerInfo`, and the `traits` decorator. The helper modules now delegate to analyzer implementations under `dryml.code.algorithms` or re-export core semantic names for compatibility.

## Problem Statement

Dispatch, probes, annotations, and later analyzers need shared answers about Python targets. Those answers should not be duplicated inside dispatch or hidden inside annotation merging.

## Guiding Principle

`dryml.code` should collect reusable algorithms that discover facts about code. It should not decide where operations run. Dispatch should consume code facts when selecting and checking candidates. Probes should be able to run code-analysis algorithms outside the orchestrator when useful.

## What Belongs in dryml.code

- Callable identity and signature inspection.
- Importability analysis.
- Source location and source text extraction.
- Source-backed fallback data for non-importable functions/classes.
- Symbol/import dependency discovery.
- Direct annotation-fragment discovery and serialized annotation requirement-resolution metadata as facts.
- Method contract facts.
- Shape hints where they are code-derived.
- AST access and method-call hints.
- Optional future dynamic trace call facts.
- Structured diagnostics.

## What Does Not Belong in dryml.code

`dryml.code` should not select environments, allocate worlds, enforce runtime policy, launch workers, or decide dispatch compatibility. Those responsibilities belong to dispatch, worlds, runtime, and provider/probe layers.

## Fact-Oriented API

Sprint 1 introduces a public `dryml.code.analyze(...)` API that returns a structured `CodeAnalysisResult`. The result contains facts and diagnostics rather than a launch decision. Facts are serializable through `to_data()` so dispatch and future probes can persist or pass them between processes.

Core public types:

- `CodeTargetSpec`: serializable target representation. It records descriptive target kind, optional import path, optional source spec, optional method metadata, and JSON-compatible metadata.
- `CodeTarget`: local analysis wrapper. It may hold live Python objects and is not used for serialized output.
- `CodeFact`: generic fact record with `kind`, `source`, and `data` fields.
- `DiagnosticFact`: structured diagnostic with severity, code, and message.
- `RequirementFact`: raw requirement/default annotation fragment fact. It preserves namespace, kind, priority, merge policy, fragment data, annotation source trace data, and serialized `RequirementResolution` data without selecting dispatch candidates.
- `CodeAnalysisContext`: analysis options such as selected algorithms, source/import permissions, annotation inclusion, method-contract inclusion, and diagnostics policy.
- `CodeAnalysisResult`: aggregate target, facts, diagnostics, `ok`, filtering helpers, and JSON-compatible serialization.

`CodeAnalysisContext.metadata` is analysis-run metadata supplied by the caller. During local analysis it is copied into normalized target metadata so serialized results can retain caller provenance, but analyzers should not interpret it as dispatch requirements or make execution decisions from it.

Example:

```python
import json
import dryml.code as code

result = code.analyze(run_training, algorithms=("callables", "source", "direct_annotations"))
callable_facts = result.facts_of_kind("callable")
json.dumps(result.to_data())
```

Built-in analyzers are registered by name:

- `callables`: callable identity, signature, and importability facts.
- `source`: source text and source-location facts.
- `ast_access`: static attribute-access and method-call-like hints.
- `symbol_capture`: `ImportRef`/`SourceSpec`-style symbol facts using `dryml.core2.symbol`.
- `direct_annotations`: raw annotation and requirement facts using the authoritative `dryml.annotations` collection/resolution APIs.
- `method_contracts`: minimal DRYML `Method` contract metadata from `dryml.core2.methods`.
- `static_calls`: opt-in conservative resolution for direct globals and direct
  methods on concretely annotated parameters. It is not in either default
  analyzer tuple.

Analyzer failures become `DiagnosticFact(error)` by default. Setting `CodeAnalysisContext(diagnostics_policy="raise")` raises a `CodeAnalysisError` instead.

Compatibility imports remain available:

- `dryml.code.callable_info.CallableInfo`
- `dryml.code.callable_info.analyze_callable`
- `dryml.code.source.SourceInfo`
- `dryml.code.source.get_source_info`
- `dryml.code.source.func_source_extract`
- `dryml.code.ast_tools.AccessCollector`
- `dryml.code.ast_tools.collect_accesses_from_source`

## Relationship to core2.symbol

`core2.symbol` already provides stable `ImportRef` and `SourceSpec` primitives. `core2.methods` provides stable method semantic primitives. `core2` must not depend on `dryml.code`; `dryml.code` may depend on `core2`. This keeps the core semantic model independent of higher-level analysis algorithms.

## Relationship to Method and Method Handles

`Method`, `Traits`, `CompilerInfo`, `BatchMode` re-export convenience, and the `traits` decorator now live under `dryml.core2.methods`. `dryml.code` re-exports these names for compatibility, but code-analysis algorithms only inspect method facts; they do not own the semantic model. Future method handle or signature semantic APIs should also live under `dryml.core2.methods` if they are introduced.

Preferred import:

```python
from dryml.core2.methods import Method, Traits, CompilerInfo, traits
```

Compatibility imports remain supported:

```python
from dryml.code import Method, Traits, CompilerInfo, traits
```

## Relationship to dispatch and probes

Dispatch should ask `dryml.code` for code facts and then apply requirement/candidate logic. Code probes should reuse the same algorithms in a lightweight `RuntimeMode.PROBE` process when orchestrator-local analysis is insufficient or risky.

Dispatch integration is intentionally deferred. Sprint 3 lets `dryml.code.algorithms.direct_annotations` delegate merge semantics to `dryml.annotations`, but the analyzer still emits facts and diagnostics only. It does not select environments, allocate worlds, enforce runtime policy, launch workers, or decide candidate compatibility.

## Sprint 9A Analysis Contract

Static analysis and future dynamic tracing share the existing `CodeAnalyzer`
protocol and `CodeAnalysisResult` model. `analyze(...)` directly runs selected
analyzers and never intentionally invokes the submitted target body. Importing an
import-path target can still execute module-level code.

`probe_target(...)` only selects execution location for the same analyzers. An
inline probe enters `RuntimeMode.PROBE` but does not imply a new OS process. A
worker process may use only analyzers installed and registered in that worker.

The Sprint 9B contract, not a currently exported API, is:

```python
dryml.code.trace(
    target,
    *,
    args=(),
    kwargs=None,
    context=None,
    policy=None,
) -> CodeAnalysisResult
```

`trace(...)` will be the invocation-bearing API and will require
`CodeAnalysisContext.allow_dynamic_execution=True`. Its invocation data is
explicit rather than hidden in metadata. Trace facts will use a distinct fact
kind, inline live notebook targets are intended use cases, and subprocess or
cross-environment tracing is not implied by this contract.

### Target and Location Support

| Requested location | Import-path target | Live non-importable target | Source-spec-only target | Timeout guarantee |
|---|---|---|---|---|
| Direct `analyze(...)` | Inline import and analysis | Inline analysis | Descriptive only; live-object analyzers report unavailable | None |
| `probe_target(..., environment=None, timeout=None)` | Inline probe runtime | Inline probe runtime | Descriptive only; no reconstruction | None |
| Current Python with finite timeout | Worker subprocess | Structured rejection | Structured reconstruction-unavailable rejection | Managed process deadline |
| `PythonExecutableSpec` | Worker subprocess | Structured rejection | Structured reconstruction-unavailable rejection | Managed process deadline when configured |
| Supported `CondaEnvironmentSpec` | Worker subprocess | Structured rejection | Structured reconstruction-unavailable rejection | Managed process deadline when configured |
| Container or remote environment | Existing unsupported diagnostic | Unsupported | Unsupported | Not applicable |

Workers independently require a stable import path, including requests sent
directly to the worker protocol. A source spec is serializable provenance data;
it does not reconstruct closures, notebook frames, bound instances, or arbitrary
dependencies. Custom analyzers registered only in the orchestrator are not
assumed to be available in a worker.

### Static Facts and Bounds

`ast_access` emits syntactic `ASTAccessFact` and `CallSiteFact` records. Every
call-site fact says `semantic_resolution="not_attempted"`; nested receivers such
as `obj.child().train()` remain partial rather than flattened through the call
result. Locations include source-relative line, file-absolute line when known,
and column offset.

`static_calls` emits one `StaticCallFact` per inspected `ast.Call` and one
`static_call_summary` fact after traversal. A resolved fact has
`status="resolved"`, `confidence="exact_static"`, and a fixed target mapping
containing only `kind`, `import_path`, `method_name`, and `subject_ref`.
Unresolved, ambiguous, and unsupported facts use `confidence="conservative_hint"`.
They are static possibilities, not runtime observations or dispatch requirements.

Supported resolution forms are plain Python functions from the analyzed
function's real globals mapping and direct methods on direct parameters with an
ordinary concrete class annotation, inspected with `inspect.getattr_static`.
String annotations, unions, generics, protocols, aliases, reassignment, nested
scopes, attribute chains, call-result receivers, properties, dynamic `getattr`,
callable instances, non-standard metaclasses, and control-flow inference do not
resolve. This restriction avoids dynamic attribute hooks while constructing facts.

Both static analyzers enforce these limits before unbounded fact expansion:

| Dimension | Limit |
|---|---:|
| UTF-8 source bytes | 1,048,576 |
| AST nodes | 100,000 |
| Call sites | 10,000 |
| Attribute/call chain components | 64 |
| Semantic-resolution diagnostics | 1,000 |
| Serialized display/reference scalars | 4,096 characters |

Bound exhaustion returns an error diagnostic with `limit_name`, `limit`, and
`observed_lower_bound`; static-call summaries then set `complete` to false.
Source unavailable, source disabled, parse failure, and no matching call remain
distinct outcomes.

## Non-Goals

- This note does not add code probes.
- This note does not add dynamic tracing or export `trace(...)`.
- This note does not implement source-spec subprocess reconstruction.
- This note does not add static-call dispatch policy or alter dispatch planning.

## Source Anchors

- `src/dryml/code/callable_info.py`
- `src/dryml/code/source.py`
- `src/dryml/code/ast_tools.py`
- `src/dryml/core2/methods/method.py`
- `src/dryml/core2/methods/traits.py`
- `src/dryml/core2/methods/compiler_info.py`
- `src/dryml/code/method.py` compatibility wrapper
- `src/dryml/code/traits.py` compatibility wrapper
- `src/dryml/core2/symbol.py`
- `src/dryml/core2/tensor_spec.py`

## Open Questions

- Should facts be dataclasses, records specs, or both?
- Which diagnostics must be JSON-compatible in Sprint 1?
- How much source-backed fallback should be accepted before probes are required?
- Should later probe/dispatch metadata use a source-text policy such as `metadata_only`, `include_text`, or `hash_only` instead of always serializing full source text?

## Follow-Up Sprints

- Sprint 1: fact-oriented code analyzer API.
- Sprint 2: Method semantic model moved to `dryml.core2.methods`.
- Sprint 5: code probe worker.
- Sprint 9: optional dynamic tracing algorithm.
